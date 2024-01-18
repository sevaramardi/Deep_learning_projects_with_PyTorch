import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load your real data from a CSV file
real_data = pd.read_csv('crop.csv')

# Extract correlation coefficients from real data
correlation_matrix = real_data.corr()
correlation_coefficient = correlation_matrix.iloc[0, 1]

# Generate synthetic data with correlation
def generate_data(num_samples, correlation_coefficient):
    np.random.seed(42)
    data = np.random.rand(num_samples, 2)  # Example: 2 features
    correlated_feature = correlation_coefficient * data[:, 0] + 0.2 * np.random.rand(num_samples)
    data[:, 1] = correlated_feature
    return data

# Define the generator model
def build_generator(latent_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(32, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(output_dim, activation='linear'))
    return model

# Define the discriminator model
def build_discriminator(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(32, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build and compile the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_model = models.Sequential()
    gan_model.add(generator)
    gan_model.add(discriminator)
    gan_model.compile(loss='binary_crossentropy', optimizer='adam')
    return gan_model

# Train the GAN
def train_gan(generator, discriminator, gan_model, data, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        fake_data = generator.predict(np.random.randn(batch_size, latent_dim))
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_data, labels_real)
        d_loss_fake = discriminator.train_on_batch(fake_data, labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.randn(batch_size, latent_dim)
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan_model.train_on_batch(noise, labels_gan)

        # Print progress
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Set parameters
num_samples = len(real_data)
latent_dim = 10
data_shape = 2  # Change this based on the number of features in your data
epochs = 5000
batch_size = 64

# Generate correlated data
real_data_array = real_data.values  # Convert DataFrame to NumPy array
real_data_array = real_data_array[:, :data_shape]  # Keep only the desired number of features
real_data_array = (real_data_array - np.min(real_data_array)) / (np.max(real_data_array) - np.min(real_data_array))  # Normalize data

# Build and compile models
generator = build_generator(latent_dim, data_shape)
discriminator = build_discriminator(data_shape)
gan_model = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan_model, real_data_array, epochs, batch_size, latent_dim)

# Generate synthetic data using the trained generator
synthetic_data = generator.predict(np.random.randn(num_samples, latent_dim))

# Denormalize synthetic data if necessary

# Display synthetic data
synthetic_df = pd.DataFrame(synthetic_data, columns=[f'Feature_{i+1}' for i in range(data_shape)])
print(synthetic_df.head())