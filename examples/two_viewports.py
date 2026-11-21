import pathlib
import tamashii as tm
from tamashii import imgui, spy

# Two views of one scene in one window. Both select and move objects, the selection is shared.

if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.absolute()
    device = spy.create_device(type=tm.vk_or_mtl_device(), enable_debug_layers=False, enable_print=False, enable_compilation_reports=False,
        include_paths=[root, root / "renderpasses", *tm.include_paths()])
    pt = tm.renderpasses.PathTracerRayQuery(device, entry_point="primal", integrator=tm.integrators.Path(emitter_samples=True, mis=tm.integrators.MIS.Balanced))

    scene = tm.Scene.cornell_box(pt.program)
    # a second camera, each view renders through its own sensor and film
    scene.sensors.entries.append(tm.sensors.Perspective.from_look_at(
        name="side", eye=[60, 480, -300], target=[278, 200, 280], up=[0, 1, 0], film=tm.films.HDRFilm(device), fov=60))
    scene.sensors.upload()

    utils = spy.Module.load_from_source(device, "utils", """import tamashii;
        void resolve(in float4 input, in float count, out float4 output) { output = float4(post::linear_to_srgb(post::tonemap_aces_filmic(input / count)).xyz, 1.0); }
    """)

    window = tm.Window(device, width=1280, height=600, title="Two Viewports")
    editor = tm.Viewport(scene.sensors[0])
    side = tm.Viewport(scene.sensors[1], controller=tm.ArcballCamera)
    views, names, samples = [editor, side], ["Editor", "Side"], [0, 0]
    # one interaction per view, on the same scene they share the selection (share_selection=False: each its own)
    context = tm.core.Context(device)
    interactions = [tm.Interaction(view, scene, context) for view in views]
    accumulate = False

    def render(ce: spy.CommandEncoder, index: int):
        view = views[index]
        if samples[index] == 1:
            view.film.clear(ce)
        # one sensor per dispatch, the path integrator would otherwise split the samples over both
        data = dict(scene.get(), sensors=scene.sensors.address_of(scene.sensors.entries.index(view.camera)), sensor_count=1)
        with ce.begin_compute_pass() as compute_pass:
            pt.render(compute_pass, data, *view.film.extent)
        utils.resolve(input=view.film.texture, count=pt["spp"] * samples[index], output=view.target, _append_to=ce)

    while window.process_events():
        if (ce := window.begin_frame()) is None:
            continue
        pt["seed"] += 1

        imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        imgui.begin("Two Viewports", None,
                    imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move
                    | imgui.WindowFlags_.no_scroll_with_mouse | imgui.WindowFlags_.no_saved_settings
                    | imgui.WindowFlags_.no_bring_to_front_on_focus)
        imgui.push_style_var(imgui.StyleVar_.cell_padding, imgui.ImVec2(0.0, 0.0))
        if imgui.begin_table("layout", 3, imgui.TableFlags_.resizable | imgui.TableFlags_.borders_inner_v):
            imgui.table_setup_column("editor", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("side", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("ui", imgui.TableColumnFlags_.width_fixed, 260.0)
            imgui.table_next_row()
            for column, view in enumerate(views):
                imgui.table_set_column_index(column)
                view.image()  # the film follows the cell
            imgui.table_set_column_index(2)
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8.0, 8.0))
            imgui.begin_child("ui", imgui.ImVec2(0.0, 0.0), imgui.ChildFlags_.always_use_window_padding)

            if (res := imgui.checkbox("Accumulate", accumulate))[0]:
                accumulate = res[1]
                pt["seed"] = 1
            if (res := imgui.checkbox("NEE", pt.integrator.emitter_samples))[0]:
                pt.integrator.emitter_samples = res[1]
            for view, interaction, name, n in zip(views, interactions, names, samples):
                imgui.separator()
                imgui.text(f"{name}: {view.width} x {view.height}, {n} samples")
                imgui.push_id(name)  # both views draw the same buttons
                view.ui()
                if (res := imgui.checkbox("Shared selection", interaction.share_selection))[0]:
                    interaction.share_selection = res[1]
                imgui.pop_id()
            imgui.separator()
            imgui.text(f"{imgui.get_io().framerate :.2f} FPS | { 1000.0 / imgui.get_io().framerate :.1f} ms")

            imgui.end_child()
            imgui.pop_style_var()
            imgui.end_table()
        imgui.pop_style_var()  # cell padding
        imgui.end()
        imgui.pop_style_var(2)  # window padding and border

        for interaction in interactions:
            interaction.frame()
        # after both frames, so an edit in one view resets the other too
        for i, interaction in enumerate(interactions):
            samples[i] = 1 if not accumulate or interaction.something_changed() else samples[i] + 1
            render(ce, i)
        window.end_frame(ce)
