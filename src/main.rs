use ml_gui::{gui::GUI, widget::WidgetType};
use ml_library::{layer::Layer, layer::LayerType::*, network::Network, convolution_params::PaddingType::*, activation::ActivationFunction::*};
use image::*;
use WidgetType::*;

fn main() {
    let layers: Vec<Layer> = vec![
        Layer::conv(5, Valid, 1, ReLU),
        Layer::conv(5, Valid, 1, ReLU),
        Layer::conv(5, Valid, 1, ReLU),
        Layer::conv(5, Valid, 1, ReLU),
        Layer::conv(5, Valid, 1, ReLU),
        Layer::dense([64, 32], Sigmoid),
        Layer::dense([32, 16], Sigmoid),
        Layer::dense([16, 9], Sigmoid),
    ];

    let mut nn = Network::new(layers, 0.04, 2);
    let sections: Vec<Vec<WidgetType>> = vec![
        vec![CostPlot], 
    ];

    // nn.load_model("assets/models/6Model");

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    app.set_epochs_per_second(20);
    app.set_cost_expiration(false, 20);
    app.set_model_name("assets/models/cnnTest");
    // sin_model(&mut app);
    conv_digit_model(app);
    // xor_model(&mut app);
    // upscale_img(&mut nn, [500, 500]);
}

pub fn xor_model(app: &mut GUI) {

    let dense_data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ]; 

    app.set_dense_data(dense_data);
    app.run();
}

pub fn dense_digit_model(mut app: GUI) {
    let mut dense_data: Vec<[Vec<f64>; 2]> = vec![];

    let img = image::open("assets/img/kanji/Tile000.png").unwrap();

    let dims = img.dimensions();

    for y in 0..dims.1 {
        for x in 0..dims.0 {
            let pixel = img.get_pixel(x, y).0;
            let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            let x_coord = x as f64 / (dims.0 - 1) as f64;
            let y_coord = y as f64 / (dims.1 - 1) as f64;
            let inputs = vec![x_coord, y_coord];
            dense_data.push([inputs, vec![(intensity as f64 / 255.0)]]);
        }
    }

    app.set_dense_data(dense_data);
    app.run();
}

pub fn conv_digit_model(mut app: GUI) {
    let mut conv_data: Vec<(Vec<Vec<f64>>, Vec<f64>)> = vec![];

    let max_nums = 9;
    for i in 0..max_nums {
        let img = image::open(format!("assets/img/mnist/mnist_{}.png", i)).unwrap();

        let dims = img.dimensions();

        let mut inputs = vec![];

        for y in 0..dims.1 {
            let mut row = vec![];
            for x in 0..dims.0 {
                let pixel = img.get_pixel(x, y).0;
                let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
                row.push(intensity as f64);
            }
            inputs.push(row);
        }
        let mut output = vec![0.0; max_nums];
        output[i] = 1.0;
        conv_data.push((inputs, output));
    }

    app.set_conv_data(conv_data);
    app.run();
}

pub fn sin_model(app: &mut GUI) {

    app.x_range = [-10.0, 10.0];

    let mut func_outputs = vec![];
    let increment = 0.1;
    let mut counter = app.x_range[0];

    while counter < app.x_range[1] {
        func_outputs.push([vec![counter.clone()], vec![func(&counter)]]);
        counter += increment;
    }

    app.set_dense_data(func_outputs);
    app.run()
}

fn func(x: &f64) -> f64 {
    x.sin()
}

fn upscale_img(nn: &mut Network, dims: [u32; 2]) {

    // let output_image = ImageBuffer::from_fn(dims[0], dims[1], |x, y| {
    //     let x_coord = x as f64 / (dims[0] - 1) as f64;
    //     let y_coord = y as f64 / (dims[1] - 1) as f64;
    //     let inputs = vec![x_coord, y_coord];
    //     let pix = (nn.forward(inputs)[0] * 255.0) as u8;
    //     Rgba([pix, pix, pix, 255]) // Varying colors in a gradient
    // });
            

    // let _  = output_image.save("Output.png").unwrap();
    // println!("Saved Image");
}