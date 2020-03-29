import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib as  plt
import seaborn as sns
import csv



CSVFILE = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/TF_based_models/TF_based_models/confusion_matrices/predictions_conv_41_classes.csv'
test_df = pd.read_csv(CSVFILE)

actualValue = test_df['label']
predictedValue = test_df['y_pred']

# actualValue = actualValue.values.argmax(axis=1)
# predictedValue  =predictedValue.values.argmax(axis=1)

cmt = confusion_matrix(actualValue, predictedValue)

print(cmt)


#%%

ax= plt.pyplot.subplot()
sns.heatmap(cmt, vmin=0, vmax=100, annot=False, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
#ax.xaxis.set_ticklabels(['ham', 'spam'], fontsize = 15)
ax.xaxis.tick_top()

ax.set_ylabel('True', fontsize=20)
#ax.yaxis.set_ticklabels(['spam', 'ham'], fontsize = 15)
plt.pyplot.show()

#%%

report = classification_report(actualValue, predictedValue)
print(report)


       