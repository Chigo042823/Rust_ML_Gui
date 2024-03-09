use ml_gui::{gui::GUI, section::Section, widget::WidgetType};
use ml_library::{layer::{DenseLayer, Layer}, network::Network};
use image::*;

fn main() {
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(DenseLayer::new(2, 7, ml_library::activation::ActivationFunction::Sigmoid)),
        Box::new(DenseLayer::new(7, 7, ml_library::activation::ActivationFunction::Sigmoid)),
        Box::new(DenseLayer::new(7, 1, ml_library::activation::ActivationFunction::Sigmoid)),
    ];
    let nn = Network::new(layers, 0.8, 1);
    let sections = vec![
        vec![WidgetType::CostPlot],
    ];

    let xor_data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ]; 

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    // app.set_data(xor_data);
    digit_model(app);
}

pub fn digit_model(mut app: GUI) {
    let mut data: Vec<[Vec<f64>; 2]> = vec![];

    let img = image::open("mnist_8.png").unwrap();

    for y in 0..img.dimensions().1 {
        for x in 0..img.dimensions().0 {
            let intensity;
            let pixel = img.get_pixel(x, y).0;
            intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            data.push([vec![x as f64, y as f64], vec![(intensity as f64 / 255 as f64)]]);
        }
    }

    // println!("-----------------------------");
    app.set_data(data);
    app.run();

}

