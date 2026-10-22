"""
Slangpy renderer for Dear ImGui, the platform side is imgui.impl_sdl3
"""

from typing import Sequence

import imgui
import numpy as np
from tamashii._spy import spy
from imgui import implot


def set_imgui_style():
    style = imgui.get_style()

    style.colors[imgui.Col_.text]                  = imgui.ImVec4(0.92, 0.93, 0.94, 1.00)
    style.colors[imgui.Col_.text_disabled]         = imgui.ImVec4(0.50, 0.52, 0.54, 1.00)
    style.colors[imgui.Col_.window_bg]             = imgui.ImVec4(0.14, 0.14, 0.16, 1.00)
    style.colors[imgui.Col_.child_bg]              = imgui.ImVec4(0.16, 0.16, 0.18, 1.00)
    style.colors[imgui.Col_.popup_bg]              = imgui.ImVec4(0.18, 0.18, 0.20, 1.00)
    style.colors[imgui.Col_.border]                = imgui.ImVec4(0.28, 0.29, 0.30, 0.60)
    style.colors[imgui.Col_.border_shadow]         = imgui.ImVec4(0.00, 0.00, 0.00, 0.00)
    style.colors[imgui.Col_.frame_bg]              = imgui.ImVec4(0.20, 0.22, 0.24, 1.00)
    style.colors[imgui.Col_.frame_bg_hovered]      = imgui.ImVec4(0.22, 0.24, 0.26, 1.00)
    style.colors[imgui.Col_.frame_bg_active]       = imgui.ImVec4(0.24, 0.26, 0.28, 1.00)
    style.colors[imgui.Col_.title_bg]              = imgui.ImVec4(0.14, 0.14, 0.16, 1.00)
    style.colors[imgui.Col_.title_bg_active]       = imgui.ImVec4(0.16, 0.16, 0.18, 1.00)
    style.colors[imgui.Col_.title_bg_collapsed]    = imgui.ImVec4(0.14, 0.14, 0.16, 1.00)
    style.colors[imgui.Col_.menu_bar_bg]           = imgui.ImVec4(0.20, 0.20, 0.22, 1.00)
    style.colors[imgui.Col_.scrollbar_bg]          = imgui.ImVec4(0.16, 0.16, 0.18, 1.00)
    style.colors[imgui.Col_.scrollbar_grab]        = imgui.ImVec4(0.24, 0.26, 0.28, 1.00)
    style.colors[imgui.Col_.scrollbar_grab_hovered] = imgui.ImVec4(0.28, 0.30, 0.32, 1.00)
    style.colors[imgui.Col_.scrollbar_grab_active] = imgui.ImVec4(0.32, 0.34, 0.36, 1.00)
    style.colors[imgui.Col_.check_mark]            = imgui.ImVec4(0.46, 0.56, 0.66, 1.00)
    style.colors[imgui.Col_.slider_grab]           = imgui.ImVec4(0.36, 0.46, 0.56, 1.00)
    style.colors[imgui.Col_.slider_grab_active]    = imgui.ImVec4(0.40, 0.50, 0.60, 1.00)
    style.colors[imgui.Col_.button]                = imgui.ImVec4(0.24, 0.34, 0.44, 1.00)
    style.colors[imgui.Col_.button_hovered]        = imgui.ImVec4(0.28, 0.38, 0.48, 1.00)
    style.colors[imgui.Col_.button_active]         = imgui.ImVec4(0.32, 0.42, 0.52, 1.00)
    style.colors[imgui.Col_.header]                = imgui.ImVec4(0.24, 0.34, 0.44, 1.00)
    style.colors[imgui.Col_.header_hovered]        = imgui.ImVec4(0.28, 0.38, 0.48, 1.00)
    style.colors[imgui.Col_.header_active]         = imgui.ImVec4(0.32, 0.42, 0.52, 1.00)
    style.colors[imgui.Col_.separator]             = imgui.ImVec4(0.28, 0.29, 0.30, 1.00)
    style.colors[imgui.Col_.separator_hovered]     = imgui.ImVec4(0.46, 0.56, 0.66, 1.00)
    style.colors[imgui.Col_.separator_active]      = imgui.ImVec4(0.46, 0.56, 0.66, 1.00)
    style.colors[imgui.Col_.resize_grip]           = imgui.ImVec4(0.36, 0.46, 0.56, 0.30)
    style.colors[imgui.Col_.resize_grip_hovered]   = imgui.ImVec4(0.40, 0.50, 0.60, 0.40)
    style.colors[imgui.Col_.resize_grip_active]    = imgui.ImVec4(0.44, 0.54, 0.64, 0.50)
    style.colors[imgui.Col_.tab]                   = imgui.ImVec4(0.20, 0.22, 0.24, 1.00)
    style.colors[imgui.Col_.tab_hovered]           = imgui.ImVec4(0.28, 0.38, 0.48, 1.00)
    style.colors[imgui.Col_.tab_selected]          = imgui.ImVec4(0.24, 0.34, 0.44, 1.00)
    style.colors[imgui.Col_.tab_dimmed]            = imgui.ImVec4(0.20, 0.22, 0.24, 1.00)
    style.colors[imgui.Col_.tab_dimmed_selected]   = imgui.ImVec4(0.24, 0.34, 0.44, 1.00)
    style.colors[imgui.Col_.plot_lines]            = imgui.ImVec4(0.46, 0.56, 0.66, 1.00)
    style.colors[imgui.Col_.plot_lines_hovered]    = imgui.ImVec4(0.46, 0.56, 0.66, 1.00)
    style.colors[imgui.Col_.plot_histogram]        = imgui.ImVec4(0.36, 0.46, 0.56, 1.00)
    style.colors[imgui.Col_.plot_histogram_hovered] = imgui.ImVec4(0.40, 0.50, 0.60, 1.00)
    style.colors[imgui.Col_.table_header_bg]       = imgui.ImVec4(0.20, 0.22, 0.24, 1.00)
    style.colors[imgui.Col_.table_border_strong]   = imgui.ImVec4(0.28, 0.29, 0.30, 1.00)
    style.colors[imgui.Col_.table_border_light]    = imgui.ImVec4(0.24, 0.25, 0.26, 1.00)
    style.colors[imgui.Col_.table_row_bg]          = imgui.ImVec4(0.20, 0.22, 0.24, 1.00)
    style.colors[imgui.Col_.table_row_bg_alt]      = imgui.ImVec4(0.22, 0.24, 0.26, 1.00)
    style.colors[imgui.Col_.text_selected_bg]      = imgui.ImVec4(0.24, 0.34, 0.44, 0.35)
    style.colors[imgui.Col_.drag_drop_target]      = imgui.ImVec4(0.46, 0.56, 0.66, 0.90)
    style.colors[imgui.Col_.nav_cursor]            = imgui.ImVec4(0.46, 0.56, 0.66, 1.00)
    style.colors[imgui.Col_.nav_windowing_highlight] = imgui.ImVec4(1.00, 1.00, 1.00, 0.70)
    style.colors[imgui.Col_.nav_windowing_dim_bg]  = imgui.ImVec4(0.80, 0.80, 0.80, 0.20)
    style.colors[imgui.Col_.modal_window_dim_bg]   = imgui.ImVec4(0.80, 0.80, 0.80, 0.35)

    style.window_padding      = imgui.ImVec2(8.0, 8.0)
    style.frame_padding       = imgui.ImVec2(5.0, 2.0)
    style.cell_padding        = imgui.ImVec2(6.0, 6.0)
    style.item_spacing        = imgui.ImVec2(6.0, 6.0)
    style.item_inner_spacing  = imgui.ImVec2(6.0, 6.0)
    style.touch_extra_padding = imgui.ImVec2(0.0, 0.0)
    style.indent_spacing      = 25.0
    style.scrollbar_size      = 11.0
    style.grab_min_size       = 10.0
    style.window_border_size  = 1.0
    style.child_border_size   = 1.0
    style.popup_border_size   = 1.0
    style.frame_border_size   = 1.0
    style.tab_border_size     = 1.0
    style.window_rounding     = 7.0
    style.child_rounding      = 4.0
    style.frame_rounding      = 3.0
    style.popup_rounding      = 4.0
    style.scrollbar_rounding  = 9.0
    style.grab_rounding       = 3.0
    style.log_slider_deadzone = 4.0
    style.tab_rounding        = 4.0

    style = implot.get_style()
    style.plot_padding = imgui.ImVec2(6.0, 8.0)

class ImguiAdapter:
    """Draws imgui with slangpy, into a target of the given format."""

    _next_id = 0  # texture ids are unique across adapters, so one window cannot draw another's

    def __init__(self, format: spy.Format, device: spy.Device) -> None:
        if not imgui.get_current_context():
            raise RuntimeError("No imgui context: call imgui.create_context() or imgui.set_current_context() first.")
        set_imgui_style()

        self.device = device
        self._textures: dict[int, tuple[spy.Texture, spy.Sampler]] = {}
        self.vbo: spy.Buffer | None = None
        self.ibo: spy.Buffer | None = None
        self.index_format = spy.IndexFormat.uint16 if imgui.INDEX_SIZE == 2 else spy.IndexFormat.uint32

        # ImDrawVert: float2 pos, float2 uv, uint32 rgba
        self.input_layout = device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0, "format": spy.Format.rg32_float},
                {"semantic_name": "TEXCOORD", "semantic_index": 0, "format": spy.Format.rg32_float, "offset": 8},
                {"semantic_name": "COLOR", "semantic_index": 0, "format": spy.Format.rgba8_unorm, "offset": 16},
            ],
            vertex_streams=[{"stride": imgui.VERTEX_SIZE}],
        )
        targets: Sequence[spy.ColorTargetDescParam] = [{
            "format": format,
            "enable_blend": True,
            "color": spy.AspectBlendDesc({
                "src_factor": spy.BlendFactor.src_alpha,
                "dst_factor": spy.BlendFactor.inv_src_alpha,
                "op": spy.BlendOp.add,
            }),
        }]
        self.program = device.load_program("imgui_renderer.slang", ["vertexMain", "fragmentMain"])
        self.pipeline = device.create_render_pipeline(program=self.program, input_layout=self.input_layout, targets=targets)

        self.io: imgui.IO = imgui.get_io()
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_vtx_offset  # honors ImDrawCmd::VtxOffset, for large meshes
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures  # creates imgui's textures on request

    def _add_texture(self, texture: spy.Texture, sampler: spy.Sampler | None = None) -> int:
        ImguiAdapter._next_id += 1
        self._textures[ImguiAdapter._next_id] = (texture, sampler or self.device.create_sampler())
        return ImguiAdapter._next_id

    def register_texture(self, texture: spy.Texture, sampler: spy.Sampler | None = None) -> imgui.ImTextureRef:
        """Bind an application-owned texture (e.g. a render result) so imgui.image() can draw it.
        ImGui blends with src_alpha, so write an alpha of 1.0. Release with unregister_texture()."""
        return imgui.ImTextureRef(self._add_texture(texture, sampler))

    def unregister_texture(self, texture: imgui.ImTextureRef) -> None:
        self._textures.pop(texture.get_tex_id(), None)

    def _update_textures(self, draw_data: imgui.ImDrawData) -> None:
        """Create, update and destroy imgui's own textures (the font atlas), as it requests."""
        for tex in draw_data.textures or ():
            if tex.status in (imgui.ImTextureStatus.want_create, imgui.ImTextureStatus.want_updates):
                self._textures.pop(tex.get_tex_id(), None)  # updates re-upload the whole texture
                tex.set_tex_id(self._add_texture(self.device.create_texture(
                    type=spy.TextureType.texture_2d,
                    format=spy.Format.rgba8_unorm,
                    width=tex.width,
                    height=tex.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="imgui_texture",
                    data=tex.get_pixels_array(),
                )))
                tex.set_status(imgui.ImTextureStatus.ok)
            elif tex.status == imgui.ImTextureStatus.want_destroy and tex.unused_frames > 0:
                self._textures.pop(tex.get_tex_id(), None)
                tex.set_tex_id(0)
                tex.set_status(imgui.ImTextureStatus.destroyed)

    def _upload_geometry(self, command_encoder: spy.CommandEncoder, draw_data: imgui.ImDrawData) -> None:
        """All draw lists into one vertex and one index buffer, grown as needed."""
        vbo_size = max(draw_data.total_vtx_count * imgui.VERTEX_SIZE, 1)
        ibo_size = max(draw_data.total_idx_count * imgui.INDEX_SIZE, 1)
        if self.vbo is None or self.vbo.size < vbo_size:
            self.vbo = self.device.create_buffer(
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource, label="imgui_vertex_buffer", size=vbo_size)
        if self.ibo is None or self.ibo.size < ibo_size:
            self.ibo = self.device.create_buffer(
                usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource, label="imgui_index_buffer", size=ibo_size)

        vtx_offset = idx_offset = 0
        for cmds in draw_data.cmd_lists:
            vtx, idx = cmds.vtx_buffer_array(), cmds.idx_buffer_array()
            command_encoder.upload_buffer_data(self.vbo, vtx_offset, vtx)
            command_encoder.upload_buffer_data(self.ibo, idx_offset, idx)
            vtx_offset += vtx.nbytes
            idx_offset += idx.nbytes

    def render(self, command_encoder: spy.CommandEncoder, target: spy.Texture, draw_data: imgui.ImDrawData) -> None:
        """Draw this frame's imgui draw data over `target`."""
        self._update_textures(draw_data)
        self._upload_geometry(command_encoder, draw_data)

        width, height = self.io.display_size.x, self.io.display_size.y
        proj = np.array([[2.0 / width, 0.0, 0.0, -1.0],
                         [0.0, -2.0 / height, 0.0, 1.0],
                         [0.0, 0.0, -1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
        scale = self.io.display_framebuffer_scale
        draw_data.scale_clip_rects(imgui.ImVec2(*scale))  # clip rects to framebuffer pixels
        fb_width, fb_height = int(width * scale.x), int(height * scale.y)

        with command_encoder.begin_render_pass({"color_attachments": [
            {"view": target.create_view({}), "load_op": spy.LoadOp.load, "store_op": spy.StoreOp.store}
        ]}) as pass_encoder:
            vtx_offset = idx_offset = 0  # of the current draw list, in vertices and indices
            for cmds in draw_data.cmd_lists:
                for cmd in cmds.cmd_buffer:
                    if (entry := self._textures.get(cmd.get_tex_id())) is None:
                        raise ValueError(f"Texture {cmd.get_tex_id()} is not registered with this ImguiAdapter "
                                         "(a texture registered with another window cannot be drawn here).")
                    texture, sampler = entry
                    uniforms = spy.ShaderCursor(pass_encoder.bind_pipeline(self.pipeline))["uniforms"]
                    uniforms["proj"].write(proj)
                    uniforms["texture"].write(texture)
                    uniforms["sampler"].write(sampler)

                    clip = cmd.clip_rect
                    pass_encoder.set_render_state({
                        "viewports": [spy.Viewport.from_size(target.width, target.height)],
                        "scissor_rects": [spy.ScissorRect({
                            "min_x": int(max(clip.x, 0)), "min_y": int(max(clip.y, 0)),
                            "max_x": int(min(clip.z, fb_width)), "max_y": int(min(clip.w, fb_height)),
                        })],
                        "vertex_buffers": [self.vbo],
                        "index_buffer": self.ibo,
                        "index_format": self.index_format,
                    })
                    pass_encoder.draw_indexed({
                        "vertex_count": cmd.elem_count,
                        "start_index_location": idx_offset + cmd.idx_offset,
                        "start_vertex_location": vtx_offset + cmd.vtx_offset,
                    })
                vtx_offset += cmds.vtx_buffer.size()
                idx_offset += cmds.idx_buffer.size()
