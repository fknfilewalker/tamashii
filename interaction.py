import atexit
import ctypes
import itertools
import math
import sys
import weakref
from abc import ABC, abstractmethod

import imgui
import numpy as np
import sdl3
from tamashii._spy import spy
from imgui import impl_sdl3, implot
from imgui import imguizmo as gizmo

import tamashii.core as core
import tamashii.emitters
import tamashii.scene
import tamashii.sensors
import tamashii.shapes
from .imgui_adapter import ImguiAdapter

_FLY_KEYS = ((imgui.Key.w, (0, 0, -1)), (imgui.Key.s, (0, 0, 1)), (imgui.Key.a, (-1, 0, 0)),
             (imgui.Key.d, (1, 0, 0)), (imgui.Key.e, (0, 1, 0)), (imgui.Key.q, (0, -1, 0)))

def _fly_input() -> spy.float3:
    """Camera space direction from WASD (plane) and E, Q (up, down)."""
    move = spy.float3(0.0, 0.0, 0.0)
    for key, direction in _FLY_KEYS:
        if imgui.is_key_down(key):
            move += spy.float3(*direction)
    return move

def _turn(q: spy.math.quatf, dx: float, dy: float, sensitivity: float) -> spy.math.quatf:
    """Yaw `q` around world up by `dx` and pitch around its own x axis by `dy`, in pixels."""
    q = spy.math.mul(spy.math.quat_from_angle_axis(-dx * sensitivity, spy.float3(0.0, 1.0, 0.0)), q)
    q = spy.math.mul(q, spy.math.quat_from_angle_axis(-dy * sensitivity, spy.float3(1.0, 0.0, 0.0)))
    return spy.math.normalize(q)

class InteractiveCamera(ABC):
    """Drives a camera from imgui input, `process_inputs()` returns whether it moved."""

    def __init__(self, camera: tamashii.sensors.Camera):
        self.camera = camera
        self.viewport: 'Viewport | None' = None  # set by Viewport, mouse input is relative to it
        self.trackpad = sys.platform == "darwin"  # scroll gestures instead of mouse buttons
        self.got_updated = False

    @abstractmethod
    def process_inputs(self) -> bool:
        pass

    def _look_at(self, eye: spy.float3, target: spy.float3, up: spy.float3):
        self.camera.look_at(eye, target, up)
        self.got_updated = True

class FpsCamera(InteractiveCamera):
    """Free flight: middle mouse looks around, WASDQE moves, the wheel sets the speed."""

    def __init__(self, camera: tamashii.sensors.Camera):
        super().__init__(camera)
        self.q = spy.math.normalize(spy.math.quat_from_look_at(spy.math.normalize(camera.target - camera.eye), spy.float3(0, 1, 0)))
        self.e = camera.eye
        self.move_speed = 10.0
        self._update()

    def _update(self):
        self._look_at(self.e, self.e + spy.math.mul(self.q, spy.float3(0.0, 0.0, -1.0)), spy.math.mul(self.q, spy.float3(0.0, 1.0, 0.0)))

    def process_inputs(self):
        self.got_updated = False
        io = imgui.get_io()
        if imgui.is_mouse_dragging(2):
            self.q = _turn(self.q, io.mouse_delta.x, io.mouse_delta.y, 0.002)
            self._update()
        if io.mouse_wheel != 0:
            self.move_speed = max(0.1, self.move_speed + io.mouse_wheel)
        move = _fly_input()
        if spy.math.length(move) > 0.0:
            self.e += spy.math.mul(self.q, move) * io.delta_time * self.move_speed
            self._update()
        return self.got_updated

class OrbitCamera(InteractiveCamera):
    """Orbits a target. Mouse: middle button rotates, right button pans, the wheel zooms.
    Trackpad: scroll rotates, shift+scroll pans, alt+scroll zooms. WASDQE moves the target."""

    def __init__(self, camera: tamashii.sensors.Camera):
        super().__init__(camera)
        self.q = spy.math.normalize(spy.math.quat_from_look_at(spy.math.normalize(camera.target - camera.eye), camera.up))
        self.c = camera.target
        self.d = spy.math.length(camera.target - camera.eye)
        self._update()

    def _update(self):
        self._look_at(spy.math.mul(self.q, spy.float3(0.0, 0.0, self.d)) + self.c, self.c, spy.math.mul(self.q, spy.float3(0.0, 1.0, 0.0)))

    @abstractmethod
    def _orbit(self, delta: spy.float2, uv: spy.float2):
        """Rotate for a pointer move of `delta` pixels ending at viewport coordinate `uv`."""

    def _size(self) -> spy.float2:
        return self.viewport.size if self.viewport is not None else spy.float2(1.0)

    def _pan(self, delta: spy.float2):
        self.c += spy.math.mul(self.q, spy.float3(-delta.x, delta.y, 0.0)) * (max(self.d, 1.0) * 0.002)
        self._update()

    def _zoom(self, wheel: float):
        self.d = max(0.1, self.d - wheel * 0.3)
        self._update()

    def process_inputs(self):
        self.got_updated = False
        io = imgui.get_io()
        if self.trackpad:
            if io.mouse_wheel_h != 0 or io.mouse_wheel != 0:
                delta = spy.float2(io.mouse_wheel_h, io.mouse_wheel) * -20.0
                if imgui.is_key_down(imgui.Key.left_shift):
                    self._pan(delta)
                elif imgui.is_key_down(imgui.Key.left_alt):
                    self._zoom(io.mouse_wheel)
                else:
                    self._orbit(delta, spy.float2(0.5) + delta / self._size())  # scrolling away from the center
        else:
            delta = spy.float2(io.mouse_delta.x, io.mouse_delta.y)
            if imgui.is_mouse_dragging(2):
                self._orbit(delta, self.viewport.sample_uv() if self.viewport is not None else spy.float2(0.5))
            elif imgui.is_mouse_dragging(1):
                self._pan(delta)
            if io.mouse_wheel != 0:
                self._zoom(io.mouse_wheel)
        move = _fly_input()
        if spy.math.length(move) > 0.0:
            self.c += spy.math.mul(self.q, spy.math.normalize(move)) * io.delta_time * max(self.d, 1.0)
            self._update()
        return self.got_updated

class ArcballCamera(OrbitCamera):
    """Orbit by dragging a virtual sphere laid over the image."""

    @staticmethod
    def _to_sphere(uv: spy.float2) -> spy.float3:
        x, y = uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0
        r2 = x * x + y * y
        if r2 > 1.0:
            return spy.float3(x, y, 0.0) / math.sqrt(r2)
        return spy.float3(x, y, math.sqrt(1.0 - r2))

    def _orbit(self, delta: spy.float2, uv: spy.float2):
        a, b = self._to_sphere(uv - delta / self._size()), self._to_sphere(uv)
        axis = spy.math.cross(b, a)
        if spy.math.length(axis) > 1e-5:
            self.q = spy.math.normalize(spy.math.mul(self.q, spy.math.quatf(axis.x, axis.y, axis.z, spy.math.dot(a, b))))
        self._update()

class EditorCamera(OrbitCamera):
    """Orbit with yaw and pitch, the horizon stays level."""

    def _orbit(self, delta: spy.float2, uv: spy.float2):
        self.q = _turn(self.q, delta.x, delta.y, 0.005)
        self._update()

def _sdl_init():
    if not sdl3.SDL_WasInit(sdl3.SDL_INIT_VIDEO):
        if sys.platform.startswith("linux"):  # slangpy takes X11 handles only
            sdl3.SDL_SetHint(sdl3.SDL_HINT_VIDEO_DRIVER, b"x11")
        if not sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO):
            raise RuntimeError(f"SDL_Init failed: {sdl3.SDL_GetError().decode()}")

def _native_handle(window) -> spy.WindowHandle:
    props = sdl3.SDL_GetWindowProperties(window)
    if sys.platform == "darwin":
        return spy.WindowHandle(sdl3.SDL_GetPointerProperty(props, sdl3.SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, None))
    if sys.platform == "win32":
        return spy.WindowHandle(sdl3.SDL_GetPointerProperty(props, sdl3.SDL_PROP_WINDOW_WIN32_HWND_POINTER, None))
    return spy.WindowHandle(sdl3.SDL_GetPointerProperty(props, sdl3.SDL_PROP_WINDOW_X11_DISPLAY_POINTER, None),
                            sdl3.SDL_GetNumberProperty(props, sdl3.SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0))

_dialogs = []  # callbacks and filters of open dialogs, SDL needs them until the callback ran

def _open_file_dialog(on_pick, filters: 'list[tuple[str, str]]'):
    """SDL's native open dialog; tkinter's aborts inside SDL's app on macOS. `on_pick(path)`
    runs once a file is chosen, filters are (name, "xml;gltf") pairs."""
    c_filters = (sdl3.SDL_DialogFileFilter * len(filters))(
        *[sdl3.SDL_DialogFileFilter(name.encode(), pattern.encode()) for name, pattern in filters])
    def callback(userdata, files, filter):
        if files and files[0]:  # NULL on error, empty list on cancel
            on_pick(files[0].decode())
        _dialogs.remove(keep)
    keep = (sdl3.SDL_DialogFileCallback(callback), c_filters)
    _dialogs.append(keep)
    window = Window.current()
    sdl3.SDL_ShowOpenFileDialog(keep[0], None, window.window if window else None, c_filters, len(filters), None, False)

# events that carry a windowID, all at the offset of SDL_WindowEvent's; the others go to every window
_WINDOW_EVENTS = (sdl3.SDL_EVENT_KEY_DOWN, sdl3.SDL_EVENT_KEY_UP, sdl3.SDL_EVENT_TEXT_INPUT, sdl3.SDL_EVENT_TEXT_EDITING,
                  sdl3.SDL_EVENT_MOUSE_MOTION, sdl3.SDL_EVENT_MOUSE_BUTTON_DOWN, sdl3.SDL_EVENT_MOUSE_BUTTON_UP,
                  sdl3.SDL_EVENT_MOUSE_WHEEL)

class Window:
    """An SDL window with its slangpy surface and its own imgui context."""
    # weak: a strong ref would keep the device alive into interpreter shutdown, which aborts
    _current: 'weakref.ref[Window] | None' = None
    _windows: 'dict[int, weakref.ref[Window]]' = {}  # by SDL window ID, to route events

    @staticmethod
    def current() -> 'Window | None':
        """The window imgui points at, see `make_current()`."""
        return Window._current() if Window._current is not None else None

    def __init__(self, device: spy.Device, width: int = 600, height: int = 600, title: str = "Window", scale: float = 1.0):
        _sdl_init()
        self.window = sdl3.SDL_CreateWindow(title.encode(), width, height, sdl3.SDL_WINDOW_RESIZABLE)
        if not self.window:
            raise RuntimeError(f"SDL_CreateWindow failed: {sdl3.SDL_GetError().decode()}")
        Window._windows[sdl3.SDL_GetWindowID(self.window)] = weakref.ref(self)
        self.width, self.height = width, height
        self.device = device
        self.scale = scale
        self._should_close = False

        self.surface = self.device.create_surface(_native_handle(self.window))
        self.surface_format = spy.Format.rgba8_unorm if spy.Format.rgba8_unorm in self.surface.info.formats else self.surface.info.formats[0]
        self.surface_usage = spy.TextureUsage.unordered_access | spy.TextureUsage.render_target
        self.surface.configure(width=self.fb_width, height=self.fb_height, format=self.surface_format, usage=self.surface_usage)

        # imgui state (io, window stack, draw data) is per window
        self.context = imgui.create_context()
        self.implot_context = implot.create_context()
        self.make_current()  # create_context leaves a previous context current
        impl_sdl3.init_for_other(self.window)  # input, cursors, clipboard, text input
        self.ui = ImguiAdapter(self.surface_format, self.device)

    def make_current(self):
        """Point imgui at this window. Done at every entry point: imgui has one current
        context and one window's poll drains the events of all windows."""
        imgui.set_current_context(self.context)
        implot.set_current_context(self.implot_context)
        Window._current = weakref.ref(self)

    @property
    def fb_width(self):
        return int(self.width * self.scale)
    @property
    def fb_height(self):
        return int(self.height * self.scale)
    @property
    def fb_extent(self) -> spy.int2:
        return spy.int2(self.fb_width, self.fb_height)

    def _on_event(self, event: sdl3.SDL_Event):
        self.make_current()
        t = event.type
        if t == sdl3.SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            self._should_close = True
        elif t == sdl3.SDL_EVENT_WINDOW_RESIZED:
            self._on_resize(event.window.data1, event.window.data2)
        elif t == sdl3.SDL_EVENT_WINDOW_MINIMIZED:
            self._on_resize(0, 0)
        elif t == sdl3.SDL_EVENT_WINDOW_RESTORED:
            w, h = ctypes.c_int(), ctypes.c_int()
            sdl3.SDL_GetWindowSize(self.window, ctypes.byref(w), ctypes.byref(h))
            self._on_resize(w.value, h.value)
        impl_sdl3.process_event(ctypes.byref(event))

    def _on_resize(self, width: int, height: int):
        self.device.wait()
        if width > 0 and height > 0:
            self.width, self.height = width, height
            self.surface.configure(width=self.fb_width, height=self.fb_height, format=self.surface_format, usage=self.surface_usage)
        else:
            self.surface.unconfigure()

    def is_minimized(self):
        return self.surface.config is None

    def should_close(self) -> bool:
        return self._should_close

    def close(self):
        if self.window is None:
            return
        self.device.wait()
        self.surface = None  # holds the rhi device, which must be gone before slangpy unloads
        self.make_current()
        impl_sdl3.shutdown()
        implot.destroy_context(self.implot_context)
        imgui.destroy_context(self.context)
        Window._current = None
        Window._windows.pop(sdl3.SDL_GetWindowID(self.window), None)
        sdl3.SDL_DestroyWindow(self.window)
        self.window = None

    @staticmethod
    def _live_windows() -> 'list[Window]':
        return [window for ref in list(Window._windows.values()) if (window := ref()) is not None]

    @staticmethod
    def _close_all():
        for window in Window._live_windows():
            window.close()

    def process_events(self) -> bool:
        """Drain SDL's queue, routing each event to its window. False once this one should close."""
        event = sdl3.SDL_Event()
        while sdl3.SDL_PollEvent(ctypes.byref(event)):
            if event.type == sdl3.SDL_EVENT_QUIT:
                for window in Window._live_windows():
                    window._should_close = True
            elif sdl3.SDL_EVENT_WINDOW_FIRST <= event.type <= sdl3.SDL_EVENT_WINDOW_LAST or event.type in _WINDOW_EVENTS:
                ref = Window._windows.get(event.window.windowID)
                if ref is not None and (window := ref()) is not None:
                    window._on_event(event)
            else:
                for window in Window._live_windows():
                    window._on_event(event)
        self.make_current()
        return not self._should_close

    def begin_frame(self) -> spy.CommandEncoder | None:
        self.make_current()
        if self.is_minimized():
            return None
        try:
            self.target = self.surface.acquire_next_image()
        except RuntimeError:  # Metal gives no drawable when none frees up within a second: skip the frame
            print("Window: no surface image, frame skipped", file=sys.stderr)
            return None
        impl_sdl3.new_frame()
        # the surface is the window's size in points times `scale`, not SDL's pixel size
        imgui.get_io().display_framebuffer_scale = imgui.ImVec2(self.scale, self.scale)
        imgui.new_frame()
        return self.device.create_command_encoder()

    def end_frame(self, command_encoder: spy.CommandEncoder):
        self.make_current()
        imgui.render()
        self.ui.render(command_encoder, self.target, imgui.get_draw_data())
        self.device.submit_command_buffer(command_encoder.finish())
        self.surface.present()

# runs before slangpy's own atexit, which was registered at import
atexit.register(Window._close_all)

class Viewport:
    """One camera's image: the sensor, its controls and the render target. The film follows
    the drawn size, so the image is never resampled. With a `controller` the user drives the
    camera while the mouse is over the image, `controller=None` only displays. Knows nothing
    of the scene, `Interaction` adds editing on top."""

    def __init__(self, camera: tamashii.sensors.Camera,
                 controller: 'type[InteractiveCamera] | None' = EditorCamera):
        self._controller: InteractiveCamera | None = None
        self.camera = camera
        self.controller = controller

        self.anchor = spy.float2(0.0)  # screen position, set when drawn
        self.size = spy.float2(camera.film.width, camera.film.height)
        self.camera_changed = False
        self.resized = False
        self.hovered = False
        self.draw_list: imgui.ImDrawList | None = None  # of the window `image()` drew into
        self._interacting = False
        self._backend: ImguiAdapter | None = None  # imgui backend of the window, bound on first use
        self._last_size: spy.int2 | None = None  # previous frame, None adopts at once
        self._target: spy.Texture | None = None
        self._target_ref: imgui.ImTextureRef | None = None

    @property
    def camera(self) -> tamashii.sensors.Camera:
        return self._camera

    @camera.setter
    def camera(self, camera: tamashii.sensors.Camera):
        """Switching keeps the controller type, which then drives the new camera."""
        self._camera = camera
        if self._controller is not None:
            self.controller = type(self._controller)

    @property
    def controller(self) -> 'InteractiveCamera | None':
        return self._controller

    @controller.setter
    def controller(self, controller: 'type[InteractiveCamera] | None'):
        """Switch the controller by type: `viewport.controller = tm.FpsCamera`."""
        old = self._controller
        self._controller = controller(self.camera) if controller is not None else None
        if self._controller is not None:
            self._controller.viewport = self
            if old is not None:
                self._controller.trackpad = old.trackpad

    @property
    def film(self):
        return self.camera.film

    @property
    def device(self) -> spy.Device:
        return self.film.device

    @property
    def width(self) -> int:
        """Drawn width in points, like `Window.width`."""
        return int(self.size[0])
    @property
    def height(self) -> int:
        return int(self.size[1])
    @property
    def extent(self) -> spy.int2:
        return spy.int2(self.width, self.height)

    @property
    def fb_extent(self) -> spy.int2:
        """Size in framebuffer pixels, what the film is sized to."""
        scale = imgui.get_io().display_framebuffer_scale
        return spy.int2(max(1, int(self.size[0] * scale.x)), max(1, int(self.size[1] * scale.y)))

    @property
    def target(self) -> spy.Texture:
        """The texture to render into, kept at the size of the film."""
        if self._backend is None:  # bind to the window whose frame is running
            window = Window.current()
            if window is None:
                raise RuntimeError("Viewport has no imgui backend, create the Window first.")
            self._backend = window.ui
        if self._target is None or self._target.width != self.film.width or self._target.height != self.film.height:
            if self._target_ref is not None:
                self._backend.unregister_texture(self._target_ref)
            self.device.wait()  # frames in flight may still sample the old texture
            self._target = self.device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba8_unorm,
                width=self.film.width,
                height=self.film.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=f"viewport_{self.camera.name}",
            )
            # imgui blends with the alpha, so whatever fills this texture has to write it
            self._target_ref = self._backend.register_texture(self._target)
        return self._target

    @property
    def texture_ref(self) -> imgui.ImTextureRef:
        """imgui handle of `target`, to draw it with imgui.image()."""
        self.target  # allocates and registers it
        assert self._target_ref is not None
        return self._target_ref

    @property
    def input_enabled(self) -> bool:
        """Whether the camera and picking may consume input this frame."""
        # want_capture_mouse is always set over the image, so hovering is the test instead
        return (self.hovered or self._interacting) and not imgui.get_io().want_text_input

    def _track_mouse(self, hovered: bool):
        self.hovered = hovered
        pressed = any(imgui.get_io().mouse_down[button] for button in range(3))
        self._interacting = pressed and (self._interacting or hovered)  # a drag survives leaving the image

    def set_rect(self, anchor: 'spy.float2param', size: 'spy.float2param'):
        """Place the viewport by hand, `image()` does it when drawing."""
        self.anchor = spy.float2(anchor[0], anchor[1])
        self.size = spy.math.max(spy.float2(size[0], size[1]), spy.float2(1.0))
        io = imgui.get_io()
        inside = (self.anchor[0] <= io.mouse_pos.x < self.anchor[0] + self.size[0]
                  and self.anchor[1] <= io.mouse_pos.y < self.anchor[1] + self.size[1])
        self._track_mouse(inside and not io.want_capture_mouse)  # no item to hover, `image()` overrides this
        # resize the film once the size settled, not on every frame of an edge drag
        size_px = self.fb_extent
        settled = self._last_size is None or self._last_size == size_px
        self.resized = self.film.extent != size_px and settled
        self._last_size = size_px
        if self.resized:
            self.camera.resize(*size_px)

    def update(self, enabled: bool = True):
        """Once per frame after layout: camera input. `enabled=False` when something else
        owns the mouse, e.g. a gizmo."""
        driving = self.controller is not None and enabled and self.input_enabled
        self.camera_changed = bool(self.controller.process_inputs()) if driving else False

    def ui(self):
        """Controller buttons and the trackpad toggle."""
        if self.controller is None:
            return
        imgui.text(f"Camera: {self.controller.__class__.__name__}")
        if imgui.button("FPS"):
            self.controller = FpsCamera
        imgui.same_line()
        if imgui.button("Arcball"):
            self.controller = ArcballCamera
        imgui.same_line()
        if imgui.button("Editor"):
            self.controller = EditorCamera
        imgui.same_line()
        self.controller.trackpad = imgui.checkbox("Trackpad", self.controller.trackpad)[1]

    def image(self, size: 'spy.float2param | None' = None):
        """Draw the image as an imgui item, `size` defaults to the available space."""
        if size is None:
            available = imgui.get_content_region_avail()
            size = spy.float2(available.x, available.y)
        cursor = imgui.get_cursor_screen_pos()
        self.set_rect(spy.float2(cursor.x, cursor.y), size)
        imgui.image(self.texture_ref, imgui.ImVec2(*self.size))
        self.draw_list = imgui.get_window_draw_list()
        self._track_mouse(imgui.is_item_hovered())

    def sample_uv(self, position: imgui.ImVec2 | None = None) -> spy.float2:
        """Mouse (or `position`) in [0, 1] viewport coordinates, e.g. to generate a ray."""
        pos = imgui.get_io().mouse_pos if position is None else position
        return spy.float2((pos.x - self.anchor[0]) / self.size[0], (pos.y - self.anchor[1]) / self.size[1])

    def project(self, position: 'spy.float4param') -> tuple[float, float] | None:
        """World position to screen coordinates, None if behind the camera or off screen."""
        clip = spy.math.mul(spy.math.mul(spy.float4(position), self.camera.to_sensor), self.camera.to_sample)
        if clip.w < 0.001 or not (-1.0 <= clip.z / clip.w <= 1.0):
            return None
        return (self.anchor[0] + (clip.x / clip.w + 1.0) * 0.5 * self.size[0],
                self.anchor[1] + (clip.y / clip.w + 1.0) * 0.5 * self.size[1])

class _SceneEdit:
    """Edit state of a scene, shared by all interactions on it."""
    def __init__(self):
        self.selected = None  # entry, for the interactions that share the selection
        self.shapes_frame = -1  # imgui frame of the last shape edit
        self.emitters_frame = -1

class Interaction:
    """Scene editing through a viewport: the scene list, gizmos and picking. Interactions on the
    same scene share the selection, so one per viewport gives several views of it (see
    two_viewports.py), `share_selection=False` gives a view its own. The application owns the frame:

        viewport = tm.Viewport(camera)
        interaction = tm.Interaction(viewport, scene, context)
        while window.process_events():
            if (ce := window.begin_frame()) is not None:
                ...                   # layout: widgets, `viewport.image()` or `set_rect()`
                interaction.frame()   # camera input, gizmos and picking
                ...                   # record the render
                window.end_frame(ce)

    `viewport.image()` and `ui()` draw at the imgui cursor, so any layout works (see main.py).
    """
    _edits: 'weakref.WeakKeyDictionary[tamashii.scene.Scene, _SceneEdit]' = weakref.WeakKeyDictionary()
    _ids = itertools.count()
    _gizmo_frame = -1  # imgui frame ImGuizmo was begun in, shared by all interactions

    def __init__(self, viewport: Viewport, scene: 'tamashii.scene.Scene', context: core.Context,
                 share_selection: bool = True):
        self.viewport = viewport
        self.scene = scene
        self.context = context
        self._share_selection = share_selection  # off: own selection, shortcuts only while hovered
        self._selected = None  # entry, when the selection is not shared
        self.op = gizmo.OPERATION.translate
        self.mode = gizmo.MODE.world
        self._id = next(Interaction._ids)  # ImGuizmo id, a drag stays with the gizmo it started on
        self._gizmo_hovered = False  # from this gizmo's manipulate, ImGuizmo's own state is the last gizmo's
        self._picked_scene = None  # path from the Load Scene dialog

    def _pick_scene(self, path: str):
        self._picked_scene = path

    @property
    def _edit(self) -> _SceneEdit:
        return Interaction._edits.setdefault(self.scene, _SceneEdit())

    @property
    def share_selection(self) -> bool:
        return self._share_selection

    @share_selection.setter
    def share_selection(self, share: bool):
        """Switching keeps this view's selection: it goes along, or into the shared one if there is one."""
        selection = self.current_selection
        self._share_selection = share
        if selection >= 0:
            self.current_selection = selection

    @property
    def current_selection(self) -> int:
        """Index into `entries()`, -1 for none. Kept as the entry, so deleting another does not shift it."""
        selected = self._edit.selected if self.share_selection else self._selected
        return next((i for i, e in enumerate(self.entries()) if e is selected), -1)

    @current_selection.setter
    def current_selection(self, index: int):
        selected = self.entries()[index] if index >= 0 else None
        if self.share_selection:
            self._edit.selected = selected
        else:
            self._selected = selected

    @property
    def shapes_changed(self) -> bool:
        """A shape was edited this frame, through any interaction on the scene."""
        return self._edit.shapes_frame == imgui.get_frame_count()

    @property
    def emitters_changed(self) -> bool:
        return self._edit.emitters_frame == imgui.get_frame_count()

    @property
    def camera_index(self) -> int:
        """Index of the viewport camera in the scene's sensor buffer."""
        return self.scene.sensors.entries.index(self.viewport.camera)

    def frame(self):
        """Once per frame between layout and render: camera input, gizmos and picking."""
        self.viewport.update(enabled=not self.blocking)
        if self.viewport.camera_changed or self.viewport.resized:
            self.scene.sensors.upload()
        self.interact()

    def ui(self):
        """Scene widgets at the imgui cursor: load button and the scene list."""
        if imgui.button("Load Scene"):
            _open_file_dialog(self._pick_scene, [("Scene Files", "xml;glb;gltf")])
        if self._picked_scene is not None:  # loaded here, the dialog's callback may run on another thread
            file_path, self._picked_scene = self._picked_scene, None
            self.scene = tamashii.scene.Scene.from_file(self.scene.program, file_path)
            self.viewport.camera = next(s for s in self.scene.sensors.entries if isinstance(s, tamashii.sensors.Camera))

        imgui.set_next_item_width(-imgui.FLT_MIN)
        self.current_selection = imgui.list_box(
            "##Scene", self.current_selection, ["- " + e.name for e in self.entries()], height_in_items=4)[1]

    @property
    def blocking(self) -> bool:
        """A gizmo under the mouse takes the input, the camera must not move as well."""
        return self.current_selection >= 0 and self._gizmo_hovered

    def entries(self) -> list:
        """Selectable scene entries: the shapes, then the emitters without a shape."""
        return (list(self.scene.shapes.entries)
                + [e for e in self.scene.emitters.entries if not isinstance(e, tamashii.emitters.Area)])

    def interact(self):
        """Gizmo, overlays and picking for the scene seen through this viewport."""
        entries, selection = self.entries(), self.current_selection
        if selection >= 0:
            self._gizmo(entries[selection])
        else:
            self._gizmo_hovered = False
        self._draw_emitters()
        if imgui.is_mouse_clicked(0) and self.viewport.input_enabled and not self.blocking:
            self.current_selection = self._pick(self.entries())  # the gizmo may have deleted one

    def something_changed(self) -> bool:
        """This viewport's camera or size, or the scene through any interaction on it."""
        return (self.viewport.camera_changed or self.viewport.resized
                or self.shapes_changed or self.emitters_changed)

    def _gizmo(self, entry):
        """ImGuizmo for the selected entry and its keyboard shortcuts."""
        # a shared selection takes the keys anywhere, an own one only while its view is hovered
        keys = self.share_selection or self.viewport.hovered
        if keys and imgui.is_key_down(imgui.Key.r):
            self.op = gizmo.OPERATION.rotate
        if keys and imgui.is_key_down(imgui.Key.t):
            self.op = gizmo.OPERATION.translate
        if keys and imgui.is_key_down(imgui.Key.y):
            self.op = gizmo.OPERATION.scale
        if keys and imgui.is_key_down(imgui.Key.l):
            self.mode = gizmo.MODE.local
        if keys and imgui.is_key_down(imgui.Key.k):
            self.mode = gizmo.MODE.world

        # once per frame: a second begin_frame resets the hover state the other gizmo just set
        if Interaction._gizmo_frame != imgui.get_frame_count():
            Interaction._gizmo_frame = imgui.get_frame_count()
            gizmo.begin_frame()
        # ImGuizmo only takes the mouse while the window of its drawlist is hovered, so draw
        # into the one holding the image instead of its own fullscreen window
        if self.viewport.draw_list is not None:
            gizmo.set_drawlist(self.viewport.draw_list)
        gizmo.set_orthographic(False)
        gizmo.set_rect(*self.viewport.anchor, *self.viewport.size)
        gizmo.allow_axis_flip(False)

        flip_y = spy.float4x4([1,0,0,0, 0,-1,0,0, 0,0,1,0, 0,0,0,1])
        # our matrices act on row vectors, so their memory is ImGuizmo's float[16] as is; the
        # binding reads [row, column] of column-vector matrices, so it gets the transpose (a view)
        matrix = lambda m: np.ascontiguousarray(m.to_numpy(), dtype=np.float32)
        obj_matrix = matrix(entry.to_world)
        gizmo.push_id(self._id)
        changed = gizmo.manipulate(
            view=matrix(self.viewport.camera.to_sensor).T,
            projection=matrix(spy.math.mul(flip_y, self.viewport.camera.to_sample)).T,
            operation=self.op,
            mode=self.mode,
            matrix=obj_matrix.T  # edited in place, through the view
        )
        self._gizmo_hovered = gizmo.is_using() or (gizmo.is_over() and self.viewport.hovered)
        gizmo.pop_id()
        if changed:
            entry.to_world = core.Transform(obj_matrix.flatten().tolist())
            if isinstance(entry, tamashii.shapes.Shape):
                self._edit.shapes_frame = imgui.get_frame_count()
                self.scene.shapes.upload_entry(self.current_selection)
                self.scene.acceleration_structure.update_tlas(self.scene.shapes.entries)
            if isinstance(entry, tamashii.emitters.Emitter):
                self._edit.emitters_frame = imgui.get_frame_count()
                self.scene.emitters.upload()
        if keys and imgui.is_key_down(imgui.Key.escape):
            self.current_selection = -1
        if keys and imgui.is_key_pressed(imgui.Key.x):
            self._delete(entry)

    def _delete(self, entry):
        """Remove `entry` from the scene, an area light goes with its shape."""
        shapes, emitters = self.scene.shapes, self.scene.emitters
        if isinstance(entry, tamashii.shapes.Shape):
            self._edit.shapes_frame = imgui.get_frame_count()
            addr = shapes.address_of(shapes.entries.index(entry))
            shapes.entries.remove(entry)
            shapes.upload()
            self.scene.acceleration_structure.update_tlas(shapes.entries)
            emitters.entries[:] = [e for e in emitters.entries
                                   if not (isinstance(e, tamashii.emitters.Area) and e.shape == addr)]
        else:
            self._edit.emitters_frame = imgui.get_frame_count()
            emitters.entries.remove(entry)
        emitters.upload()
        self.current_selection = -1

    def _draw_emitters(self):
        """Circles around the point lights, on top of the image."""
        viewport = self.viewport
        draw_list = imgui.get_foreground_draw_list()
        color = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.85, 0.1, 0.9))
        lo, hi = viewport.anchor, viewport.anchor + viewport.size
        for emitter in self.scene.emitters.entries:
            if isinstance(emitter, tamashii.emitters.Point):
                screen = viewport.project(emitter.to_world[3])
                # the foreground is not clipped, keep other viewports' lights out of this one
                if screen is not None and lo.x <= screen[0] < hi.x and lo.y <= screen[1] < hi.y:
                    draw_list.add_circle(imgui.ImVec2(*screen), 10.0, color, 32, 2.0)

    def _pick(self, entries: list) -> int:
        """Entry under the mouse, -1 for none: the closest shape hit by a camera ray through
        the mouse, or a point light drawn within a few pixels of it and not behind that hit."""
        rs = self.context.structs[self.viewport.camera.implementation_type_name].sample_ray(
            self.scene.sensors.data_at(self.camera_index), self.viewport.sample_uv()
        )
        ray = core.Ref(rs.ray)
        bary, prim, geom, best = core.Ref(spy.float2(0.0, 0.0)), core.Ref(0), core.Ref(0), -1
        for i, entry in enumerate(self.scene.shapes.entries):
            shape_data = self.scene.shapes.data_at(i)
            os_ray = self.context.Ray.transform(core.Ref(ray.value.copy()), core.download_transform(entry.to_world_tensor, 1))
            if not self.context.AABB.intersect(core.Ref(shape_data.value['aabb']), os_ray).value:
                continue
            if self.context.structs[entry.implementation_type_name].intersect(shape_data, ray, 0, bary, prim, geom).value:
                best = i  # intersect() shortens ray.tmax to the hit
        mouse = imgui.get_io().mouse_pos
        for i, entry in enumerate(entries):
            if isinstance(entry, tamashii.emitters.Point):
                pos_w = entry.to_world[3]
                t = float(spy.math.dot(pos_w.xyz - ray.value['origin'], ray.value['direction']))
                screen = self.viewport.project(pos_w)
                if t > float(ray.value['tmax']) or screen is None:
                    continue
                if math.hypot(mouse.x - screen[0], mouse.y - screen[1]) < 10.0:
                    ray.value['tmax'] = t
                    best = i
        return best
