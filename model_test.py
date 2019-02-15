from keras.models import load_model
from scipy.stats import norm
import sys,os,cv2,csv,numpy as np

DIR='./face128_test/'

def predict_img(model, file):
    img = cv2.imread(file)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

model = load_model('model_450faces.h5')
model.summary()

filelist = os.listdir(DIR)
pred=[]
for FILE in filelist:
  PATH=DIR+FILE
  print(FILE,end=' ')
  pred.append(predict_img(model,PATH)[0][0])

ans=[]
with open('human_ratings.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    ans.append(float(row['Attractiveness label']))

print('\n\nPredicted beauty scores:\n', pred)
corr = np.corrcoef(ans[-50:], pred)[0, 1]
print('\nCorrelation: ', corr)
