import datetime
import tamashii as tm
from tamashii import imgui, spy

if __name__ == "__main__":
    device = spy.create_device(type=tm.vk_or_mtl_device(), enable_debug_layers=False, enable_print=False, enable_compilation_reports=False,
        include_paths=tm.include_paths()
    )
    pt = tm.renderpasses.PathTracerRayQuery(device, entry_point="primal", integrator=tm.integrators.Path(mis=1))

    # sphere on a ground plane, lit only by the sky; sun placed by date/time below
    sky_date = (2025, 6, 21)
    scene = tm.Scene.hosek_wilkie(pt.program, date=sky_date, hour=12.0)
    camera = scene.sensors[0]
    assert isinstance(camera, tm.sensors.Perspective)
    film = camera.film

    utils = spy.Module.load_from_source(device, "utils", """import tamashii;
        void resolve(inout float4 img, uint count) { img = float4(img.xyz / count, 1.0); }
        void resolve(in float4 input, in float count, out float4 output) { output = float4(post::tonemap_aces_filmic(post::linear_to_srgb(input / count)).xyz, 1.0); }
    """)

    samples = 0
    accumulate = False
    day_of_year = datetime.date(*sky_date).timetuple().tm_yday
    window = tm.Window(device, width=1000, height=600, title="Sky")
    viewport = tm.Viewport(camera)
    interaction = tm.Interaction(viewport, scene, tm.core.Context(device))
    while window.process_events():
        if (ce := window.begin_frame()) is not None:
            pt["seed"] += 1
            reset = False

            imgui.begin("Settings")
            interaction.ui()
            imgui.separator()
            viewport.ui()

            imgui.separator()
            imgui.text("Integrator:")
            if (res := imgui.checkbox("Accumulate", accumulate))[0]:
                accumulate = res[1]
                pt["seed"] = 1
            if (res := imgui.checkbox("NEE", pt.integrator.emitter_samples))[0]:
                pt.integrator.emitter_samples = res[1]

            # moving the sun re-cooks on the CPU and re-uploads ~180 floats
            sky = next((e for e in interaction.scene.emitters.entries
                        if isinstance(e, tm.emitters.HosekWilkie)), None)
            if sky is not None:
                imgui.separator()
                imgui.text("Sky:")
                sky_changed = False
                if (res := imgui.slider_float("Hour", sky.hour, 0.0, 24.0))[0]:
                    sky.hour, sky_changed = res[1], True
                if (res := imgui.slider_int("Day of year", day_of_year, 1, 365))[0]:
                    day_of_year, sky_changed = res[1], True
                if (res := imgui.slider_float("Latitude", sky.latitude, -66.5, 66.5))[0]:
                    sky.latitude, sky_changed = res[1], True
                if (res := imgui.slider_float("Turbidity", sky.turbidity, 1.0, 10.0))[0]:
                    sky.turbidity, sky_changed = res[1], True
                if (res := imgui.slider_float("Scale", sky.scale, 0.001, 0.2))[0]:
                    sky.scale, sky_changed = res[1], True
                date = datetime.date(sky.year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
                if sky_changed:
                    sky.configure_from_time(month=date.month, day=date.day)
                    interaction.scene.emitters.upload()
                    reset = True
                imgui.text(f"{date:%d %b}  {int(sky.hour):02d}:{int(sky.hour % 1 * 60):02d}"
                            f"   sun {sky.elevation:5.1f} deg up, {sky.azimuth:6.1f} deg az"
                            f"{'' if sky.elevation > 0.0 else '   (night)'}")

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
            samples = 1 if reset or not accumulate or interaction.something_changed() else samples + 1

            if samples == 1:
                viewport.film.clear(ce)
            with ce.begin_compute_pass() as compute_pass:
                pt.render(compute_pass, interaction.scene.get(), *viewport.film.extent)
            if viewport.film.extent == window.fb_extent:
                utils.resolve(input=viewport.film.texture, count=pt["spp"] * samples, output=window.target, _append_to=ce)
            window.end_frame(ce)
