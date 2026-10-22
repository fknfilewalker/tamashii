import tamashii as tm
from tamashii import imgui, spy

if __name__ == "__main__":
    device = spy.create_device(type=tm.vk_or_mtl_device(), enable_debug_layers=False, enable_print=False, enable_compilation_reports=False,
        include_paths=tm.include_paths()
    )
    pt = tm.renderpasses.PathTracerRayQuery(device, entry_point="primal")
    pt.settings["mis"] = 1
    pt.settings["mode"] = 0
    pt.settings["aov_mode"] = 0
    
    scene = tm.Scene.cornell_box(pt.program)
    camera = scene.sensors[0]
    assert isinstance(camera, tm.sensors.Perspective)
    film = camera.film

    utils = spy.Module.load_from_source(device, "utils", """import tamashii;
        void copy(in float4 input, out float4 output) { output = input; }
        void resolve(inout float4 img, uint count) { img = float4(img.xyz / count, 1.0); }
        void resolve(in float4 input, in float count, out float4 output) { output = float4(post::tonemap_aces_filmic(post::linear_to_srgb(input / count)).xyz, 1.0); }
    """)

    if tm.tev.check():
        pt["spp"] = 256
        def render():
            pt["seed"] += 1
            ce = device.create_command_encoder()
            with ce.begin_compute_pass() as compute_pass:
                pt.render(compute_pass, scene.get(), film.width, film.height)
            utils.resolve(img=film.texture, count=pt["spp"], _append_to=ce)
            device.wait_for_submit(device.submit_command_buffer(ce.finish()))
            device.flush_print()

        render()
        tm.tev.show("Film Texture", film.texture.to_numpy(), channel_names=["R", "G", "B", "A"])
    else:
        samples = 0
        accumulate = False
        window = tm.Window(device, width=1000, height=600, title="Path Tracer")
        viewport = tm.Viewport(camera)
        interaction = tm.Interaction(viewport, scene, tm.core.Context(device))
        while window.process_events():
            if (ce := window.begin_frame()) is not None:
                pt["seed"] += 1

                imgui.begin("Settings")
                interaction.ui()
                imgui.separator()
                viewport.ui()

                imgui.separator()
                imgui.text("Integrator:")
                if (res := imgui.checkbox("Accumulate", accumulate))[0]:
                    accumulate = res[1]
                    pt["seed"] = 1
                if (res := imgui.checkbox("NEE", pt["nee"]))[0]:
                    pt["nee"] = res[1]

                imgui.separator()
                imgui.text("Stats:")
                imgui.text(f"Window: {window.width} x {window.height}")
                imgui.text(f"Viewport: {viewport.width} x {viewport.height}")
                imgui.text(f"Framebuffer: {viewport.fb_extent.x} x {viewport.fb_extent.y}")
                imgui.text(f"Samples: {samples}")
                imgui.text(f"{imgui.get_io().framerate :.2f} FPS | { 1000.0 / imgui.get_io().framerate :.1f} ms")
                imgui.end()

                viewport.set_rect(spy.float2(0.0), spy.float2(window.width, window.height))
                interaction.frame()
                samples = 1 if not accumulate or interaction.something_changed() else samples + 1

                if samples == 1:
                    viewport.film.clear(ce)
                with ce.begin_compute_pass() as compute_pass:
                    pt.render(compute_pass, interaction.scene.get(), *viewport.film.extent)
                if viewport.film.extent == window.fb_extent:
                    utils.resolve(input=viewport.film.texture, count=pt["spp"] * samples, output=window.target, _append_to=ce)
                window.end_frame(ce)
