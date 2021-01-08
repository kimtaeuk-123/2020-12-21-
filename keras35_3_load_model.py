import numpy as np
a = np.array(range(1, 11))

from tensorflow.keras.models import load_model
model = load_model('./keras32_split1_LSTM.py')

model.summary()

