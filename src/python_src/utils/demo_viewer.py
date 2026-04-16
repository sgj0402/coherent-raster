# Copyright (c) 2026 POSTECH (Pohang University of Science and Technology) and ETRI (Electronics and Telecommunications Research Institute)
#
# This software is licensed under the MIT License.
# See the LICENSE file for details.
#
# ------------------------------
# PATENT NOTICE
# ------------------------------
# This software may implement technologies that are subject to patents
# owned by POSTECH and ETRI.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT RIGHTS ARE GRANTED
# UNDER THIS LICENSE.
#
# Commercial use of this software that requires the use of
# patented technologies may require a separate patent license
# from POSTECH and ETRI.
#
# For licensing inquiries, please contact POSTECH and ETRI.

import time
import ctypes

import glfw
from OpenGL.GL import *
import platform  # re-import after OpenGL wildcard which shadows 'platform'
import imgui
from imgui.integrations.glfw import GlfwRenderer
import torch
import viser

# Try importing CUDA components safely
try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def check_cuda_err(err):
    """Check for CUDA Driver and Runtime API errors."""
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Error: {err.value}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime Error: {err.value}")


class ZeroCopyGLViewer:
    def __init__(self, width, height, window_title="GSplat 4K Viewer"):
        
        if platform.system() == "Windows":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Enable High DPI support
            except Exception as e:
                print(f"Failed to set DPI awareness: {e}")
                
        self.width = width
        self.height = height
        self.window_title = window_title 
        
        self.use_cuda_interop = False 
        
        # Fullscreen state variables
        self.is_fullscreen = False
        self.window_pos = (100, 100)
        self.window_size = (width, height)
        self.f11_pressed = False

        if not glfw.init():
            raise Exception("GLFW init failed")
            
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)

        self.window = glfw.create_window(width, height, window_title, None, None)
        glfw.set_window_pos(self.window, 100, 100)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # V-Sync OFF

        # --- Texture & PBO Setup ---
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Optimize texture parameters for 4K
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.data_size = width * height * 3
        self.pbo_id = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.data_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # --- CUDA Interop Setup ---
        if CUDA_AVAILABLE:
            try:
                check_cuda_err(cuda.cuInit(0))
                err, self.cuda_resource = cuda.cuGraphicsGLRegisterBuffer(
                    self.pbo_id, 
                    cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
                )
                check_cuda_err(err)
                self.use_cuda_interop = True
                print("[Native Viewer] CUDA-OpenGL Interop: ENABLED (Fast Mode)")
            except RuntimeError as e:
                print(f"[Native Viewer] CUDA-OpenGL Interop: FAILED (You have to use native Windows, not WSL.) ({e})")
                print("[Native Viewer] Set your light field display as the main display in the display setting menu.")
                self.use_cuda_interop = False
        else:
            print("[Native Viewer] cuda-python not installed. Using CPU mode.")

        # --- ImGui Setup ---
        imgui.create_context()
        io = imgui.get_io()
        
        # [Note] Global font scale: 3.0~4.0 is recommended for 4K
        io.font_global_scale = 8.0  
        
        self.impl = GlfwRenderer(self.window)
        self.overlay_text = "Initializing..."

    def get_current_monitor(self):
        """Find the monitor that contains the center of the window."""
        wx, wy = glfw.get_window_pos(self.window)
        ww, wh = glfw.get_window_size(self.window)
        cx = wx + ww // 2
        cy = wy + wh // 2

        monitors = glfw.get_monitors()
        for monitor in monitors:
            mx, my = glfw.get_monitor_pos(monitor)
            mode = glfw.get_video_mode(monitor)
            mw, mh = mode.size.width, mode.size.height
            if mx <= cx < mx + mw and my <= cy < my + mh:
                return monitor
        return glfw.get_primary_monitor()

    def toggle_fullscreen(self):
        """Toggle borderless fullscreen mode."""
        if not self.is_fullscreen:
            # [Windowed -> Fullscreen]
            
            # 1. Save current position/size
            self.window_pos = glfw.get_window_pos(self.window)
            self.window_size = glfw.get_window_size(self.window)

            # 2. Get target monitor info
            monitor = self.get_current_monitor()
            mode = glfw.get_video_mode(monitor)
            mx, my = glfw.get_monitor_pos(monitor)

            # 3. Remove borders (Core)
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)

            # 4. Adjust window position/size to fill the monitor
            # Note: Set the second argument to 'None' to keep the 'windowed' attribute.
            glfw.set_window_monitor(self.window, None, mx, my, mode.size.width, mode.size.height, 0)
            
            self.is_fullscreen = True
        else:
            # [Fullscreen -> Windowed]
            
            # 1. Restore borders
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
            
            # 2. Restore original position/size
            glfw.set_window_monitor(self.window, None, self.window_pos[0], self.window_pos[1], self.window_size[0], self.window_size[1], 0)
            
            self.is_fullscreen = False

    def update(self, image_tensor: torch.Tensor, overlay_text: str = None):
        """
        Optimized Update function:
        - Validation removed (Caller must ensure valid input)
        - Window Title update removed
        - Only performs Copy & Draw
        """
        if overlay_text:
            self.overlay_text = overlay_text

        self.impl.process_inputs()

        # Handle key inputs
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        
        if glfw.get_key(self.window, glfw.KEY_F11) == glfw.PRESS:
            if not self.f11_pressed:
                self.toggle_fullscreen()
                self.f11_pressed = True
        else:
            self.f11_pressed = False

        if glfw.window_should_close(self.window):
            return False

        # Set viewport to match framebuffer size (handles HiDPI scaling)
        fb_width,fb_height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, fb_width, fb_height)

        # [Core] CUDA Copy (No Checks for Speed)
        # Assumes the caller provides a contiguous, CUDA, uint8 tensor.
        if self.use_cuda_interop:
            try:
                # 1. Map
                check_cuda_err(cuda.cuGraphicsMapResources(1, self.cuda_resource, 0))
                err, d_ptr, _ = cuda.cuGraphicsResourceGetMappedPointer(self.cuda_resource)
                check_cuda_err(err)
                
                # 2. Copy (No checks)
                src_ptr = image_tensor.data_ptr()
                check_cuda_err(cuda.cuMemcpyDtoD(d_ptr, src_ptr, self.data_size))
                
                # 3. Unmap
                check_cuda_err(cuda.cuGraphicsUnmapResources(1, self.cuda_resource, 0))
            except RuntimeError as e:
                # Fallback only on error (normally skipped)
                print(f"[Error] CUDA Copy: {e}")
                self.use_cuda_interop = False
        
        # CPU Fallback (If CUDA fails)
        if not self.use_cuda_interop:
            image_cpu = image_tensor.cpu().numpy()  # Tensor is already uint8
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, self.data_size, image_cpu)

        # Texture Update (PBO -> Texture)
        # Transfer PBO (GPU memory) contents to texture.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Draw Fullscreen Quad
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f( 1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f( 1.0,  1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0,  1.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Draw ImGui Overlay
        imgui.new_frame()
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_bg_alpha(0.5) 
        imgui.begin("Stats", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE)
        imgui.text_colored(self.overlay_text, 1.0, 1.0, 0.0, 1.0) 
        imgui.end()
        imgui.render()
        self.impl.render(imgui.get_draw_data())

        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True
    


class CustomViewerConfig:
    orbit_center_distance: float = 2.0


class CustomViewer:

    def __init__(
        self,
        server: viser.ViserServer
    ):
        self.server = server
        self.viewer_config = CustomViewerConfig()
        
        self.server.gui.set_panel_label("Custom Viewer")
        self._init_gui()

    def _init_gui(self):
        with self.server.gui.add_folder("Gsplat"):
            orbit_distance_number = self.server.gui.add_number(
                "Orbit Distance",
                initial_value=self.viewer_config.orbit_center_distance,
                min=0.1,
                max=100.0,
                step=0.1,
                hint="Distance from camera to the dynamic orbit center.",
            )

            @orbit_distance_number.on_update
            def _(_) -> None:
                self.viewer_config.orbit_center_distance = orbit_distance_number.value