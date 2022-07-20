import argparse
import os
import threading
import time

import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from slam import *


class ReconstructionWindow():
    def __init__(self, font_id, config):
        self.config = config
        self.slam = SLAM(0.03)

        self.window = gui.Application.instance.create_window(
            '3D Reconstruction', 1920, 1080)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Info tab
        tab2 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab2.add_child(self.output_info)
        tabs.add_tab('Info', tab2)

        self.panel.add_child(gui.Label('Settings'))
        self.panel.add_fixed(vspacing)
        self.panel.add_child(b)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(tabs)
        
        # Scene
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)

        threading.Thread(name='UpdateMain', target=self.update_main).start()
        self.is_done = False
        self.is_started = False
        self.is_running = False
    
    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y, rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(
            rect.get_right() - fps_panel_width,
            rect.y, fps_panel_width,
            fps_panel_height
        )

    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(self.window, self._on_start)
        self.is_running = not self.is_running

    def _on_start(self):
        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((5000, 3), dtype=np.float32))
        )
        mat = rendering.Material()
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)
        self.is_started = True

    def _on_close(self):
        self.is_done = True

    def init_render(self, depth_ref, color_ref):
        self.input_depth_image.update_image(depth_ref.colorize_depth(1000, 0, 3))
        self.input_color_image.update_image(color_ref)

        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def update_render(self, input_depth, input_color, pcd, frustum):
        self.input_depth_image.update_image(input_depth.colorize_depth(1000, 0, 3))
        self.input_color_image.update_image(input_color)

        if self.is_scene_updated:
            if pcd is not None:
                mat = rendering.Material()
                self.widget3d.scene.scene.add_geometry(
                    'points', pcd, mat)

        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.Material()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)

    def update_main(self):
        self.idx = 0
        color_dir = os.path.join(self.config['dataset'], 'rgb')
        depth_dir = os.path.join(self.config['dataset'], 'depth')
        color_t = o3d.t.io.read_image(color_dir + '/{0:05d}.jpg'.format(self.idx))
        depth_t = o3d.t.io.read_image(depth_dir + '/{0:05d}.png'.format(self.idx))
        color_raw = o3d.io.read_image(color_dir + '/{0:05d}.jpg'.format(self.idx))
        depth_raw = o3d.io.read_image(depth_dir + '/{0:05d}.png'.format(self.idx))

        gui.Application.instance.post_to_main_thread(
            self.window,
            lambda: self.init_render(depth_t, color_t)
        )

        intrinsic_matrix = o3c.Tensor(intrinsic.intrinsic_matrix)

        start = time.time()
        while not self.is_done:
            if not self.is_started or not self.is_running:
                time.sleep(0.05)
                continue

            color_t = o3d.t.io.read_image(color_dir + '/{0:05d}.jpg'.format(self.idx))
            depth_t = o3d.t.io.read_image(depth_dir + '/{0:05d}.png'.format(self.idx))
            rgbd_image = o3d.t.geometry.RGBDImage(color_t, depth_t)
            source_t = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_matrix)

            color_raw = o3d.io.read_image(color_dir + '/{0:05d}.jpg'.format(self.idx))
            depth_raw = o3d.io.read_image(depth_dir + '/{0:05d}.png'.format(self.idx))
            source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

            source = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)       

            self.slam.update(source)

            end = time.time()
            elapsed = end - start
            start = time.time()
            self.output_fps.text = 'FPS: {:.3f}'.format(1 / elapsed)

            info = 'Frame {}/{}\n\n'.format(self.idx, 2000)
            info += 'Transformation:\n{}\n'.format(
                np.array2string(self.slam.graph.nodes[-1].pose,
                    precision=3,
                    max_line_width=40,
                    suppress_small=True
                )
            )
            self.output_info.text = info
            self.is_scene_updated = True

            T = o3c.Tensor(self.slam.graph.nodes[-1].pose, dtype=o3c.Dtype.Float32)
            source_t = source_t.voxel_down_sample(0.05).transform(T)

            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render(depth_t, color_t, source_t, self.slam.frustum))

            self.idx += self.config['step']
            self.is_done = self.is_done | (self.idx >= self.config['frames'])

        time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--step', type=int, default=8)
    args = vars(parser.parse_args())
    
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = ReconstructionWindow(mono, args)
    app.run()
