use graphics::{clear, rectangle, Rectangle};
use opengl_graphics::GlGraphics;
use piston::*;
use glutin_window::*;

use crate::section::Section;

const PADDING: f64 = 10.0;

pub struct GUI {
    pub window: GlutinWindow,
    pub gl: GlGraphics,
    pub sections: Vec<Section>,
    pub cost: Vec<f64>
}

impl GUI {
    pub fn new(cost: Vec<f64>) -> Self {
        let window = WindowSettings::new("GUI", [720, 480])
        .graphics_api(OpenGL::V3_2)
        .exit_on_esc(true)
        .build()
        .unwrap();

        GUI {
            window,
            gl: GlGraphics::new(OpenGL::V3_2),
            sections: vec![],
            cost
        }
    }

    pub fn set_sections(&mut self, section_count: usize) {
        let section_width = self.window.draw_size().width / section_count as f64;

        let mut sects = vec![];
        for i in 0..section_count {
            let coords1 = [section_width * i as f64, 0.0];
            let coords = [coords1[0] + PADDING,
                coords1[1] + PADDING,
                coords1[0] - PADDING,
                coords1[1] - PADDING
            ];
            sects.push(Section::new(coords, section_width, self.window.draw_size().height, self.cost.clone()));
            sects[i].set_widgets(2);
        }
        self.sections = sects;
    }

    pub fn render(&mut self, args: RenderArgs) {
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

        self.gl.draw(args.viewport(), |ctx, gl| {
            clear(BLACK, gl);

            for i in 0..self.sections.len() {
                self.sections[i].render(ctx, gl);
            }
        });
    }

    pub fn run(&mut self) {
        let mut events = Events::new(EventSettings::new());
        while let Some(e) = events.next(&mut self.window) {
            if let Some(args) = e.render_args() {
                self.render(args);
            }

            if let Some(args) = e.update_args() {
                self.cost.push(1.0);
                println!("{:?}", self.cost);
            }
        }
    }
}