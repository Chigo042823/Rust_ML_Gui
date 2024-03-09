use glutin_window::GlutinWindow;
use ml_gui::gui::GUI;
use opengl_graphics::*;
use piston::*;


fn main() {
    let cost = vec![];
    let mut app = GUI::new(cost);
    app.set_sections(1);
    app.run();
}
