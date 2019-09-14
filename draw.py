import matplotlib.pyplot as plt

acc = [0.358, 0.483]
val_acc = [0.58, 0.83]

loss = [0.458, 0.383]
val_loss = [0.258, 0.183]

# 左上
plt.subplot(221)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# 右上
plt.subplot(222)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# 左下
plt.subplot(223)

# 右下
plt.subplot(224)


plt.show()
