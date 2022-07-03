import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


def main():
    audio = tfio.audio.AudioIOTensor('can-open-1.wav')
    audio_tensor = audio.to_tensor()[:, 0]
    audio_tensor = tf.cast(audio_tensor, tf.float32)

    # plt.plot(audio_tensor)
    # plt.show()

    print(audio_tensor)

    spectrogram = tfio.audio.spectrogram(audio_tensor, nfft=512, window=512, stride=256).numpy()
    print(spectrogram.shape)
    plt.figure()
    plt.imshow(spectrogram.T)
    plt.show()


if __name__ == '__main__':
    main()
