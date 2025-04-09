from keras import Input
from manim import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class NeuralNetworkLearning(Scene):
    def construct(self):
        # Explicitly set renderer with no audio
        # self.renderer.camera_config["use_audio"] = False

        # Create and setup the neural network
        def create_model():
            model = Sequential([
                Input(shape=(1,)),
                Dense(100, activation='tanh'),
                Dense(100, activation='tanh'),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=1e-2)
            model.compile(optimizer=optimizer, loss='mse')
            return model

        # Generate data points
        x_data = np.random.uniform(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
        x_data = np.sort(x_data)
        y_data = (np.sin(0.5 * x_data) +
                  np.sin(x_data) +
                  0.5 * np.sin(2 * x_data) +
                  np.random.uniform(-0.1, 0.1, size=x_data.shape))

        # Points for smooth curve visualization
        test_x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)

        # Function to convert data to Manim coordinates
        def to_manim_coords(x, y):
            # Scale to fit in scene and maintain aspect ratio
            scale_factor = 3
            x_scaled = x / (2 * np.pi) * scale_factor
            y_scaled = y * scale_factor / 3  # Further adjust y to fit nicely
            return list(zip(x_scaled.flatten(), y_scaled.flatten()))

        # Create the axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"color": BLUE},
        )

        # Add axes labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        y_label.shift(DOWN * 0.3)


        # Create data points
        data_coords = to_manim_coords(x_data, y_data)
        data_points = VGroup(*[Dot(axes.c2p(x, y), color=RED, radius=0.05) for x, y in data_coords])

        # Title with epoch and loss
        title = Text("Neural Network Learning", font_size=24).to_edge(UP)
        epoch_text = Text("Epoch: 0", font_size=20).to_corner(UR)
        loss_text = Text("Loss: ---", font_size=20).next_to(epoch_text, DOWN, aligned_edge=LEFT)

        # Initial scene setup
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(epoch_text),
            Write(loss_text),
        )

        self.play(Create(data_points))

        # Create initial prediction line
        model = create_model()
        initial_pred = model.predict(test_x, verbose=0)
        pred_coords = to_manim_coords(test_x, initial_pred)
        pred_line = VMobject()
        pred_line.set_points_smoothly([axes.c2p(x, y) for x, y in pred_coords])
        pred_line.set_color(BLUE)

        self.play(Create(pred_line))

        # Add legend
        legend = VGroup(
            Dot(color=RED),
            Text("True Data", font_size=16),
            Line(start=LEFT, end=RIGHT, color=BLUE),
            Text("NN Prediction", font_size=16)
        ).arrange_in_grid(rows=2, cols=2, cell_alignment=LEFT)
        legend.to_corner(DR)
        self.play(FadeIn(legend))

        # Training loop with animation updates
        num_epochs = 2001
        update_freq = 10

        for epoch in range(1, num_epochs):
            # Training step
            with tf.GradientTape() as tape:
                y_pred = model(x_data, training=True)
                loss = tf.reduce_mean(tf.square(y_pred - y_data))

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update visualization every few epochs
            if epoch % update_freq == 0:
                # Get new predictions
                new_pred = model(test_x, training=False).numpy()
                new_coords = to_manim_coords(test_x, new_pred)

                # Create new prediction line
                new_line = VMobject()
                new_line.set_points_smoothly([axes.c2p(x, y) for x, y in new_coords])
                new_line.set_color(BLUE)

                # Update epoch and loss text
                new_epoch_text = Text(f"Epoch: {epoch}", font_size=20).to_corner(UR)
                new_loss_text = Text(f"Loss: {loss.numpy():.6f}", font_size=20).next_to(new_epoch_text, DOWN,
                                                                                        aligned_edge=LEFT)

                # Animate the transition
                self.play(
                    Transform(pred_line, new_line),
                    Transform(epoch_text, new_epoch_text),
                    Transform(loss_text, new_loss_text),
                    run_time=0.05  # Faster animation
                )

        # Final state
        final_title = Text("Training Complete", font_size=24, color=GREEN).to_edge(UP).shift(UP * 0.3)
        self.play(
            Transform(title, final_title),
            Flash(pred_line, color=GREEN, flash_radius=0.3)
        )
        self.wait(2)

# When running this file, you can use these command line flags:
# manim -pql --disable_caching neural_network_learning.py NeuralNetworkLearning

# Or modify your config.py file to include:
# config.disable_caching = True
# config.media_width = "1280"
# config.media_height = "720"
# config.disable_sound = True