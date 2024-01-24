import math
import matplotlib.pyplot as plt
import tensorflow as tf

def show_images(model, img_list, labels, class_names=None, col_size=5):
    holder_size = 4
    probs = model.predict(img_list)
    pred_labels = tf.argmax(probs, axis=-1)
    
    row_size = math.ceil(len(img_list) / col_size)
    
    fig, axes = plt.subplots(row_size, col_size, figsize=(col_size*holder_size, row_size*holder_size))
    for i, img in enumerate(img_list):
        if row_size == 1:
            ax = axes[i%col_size]
        else:
            ax = axes[i//col_size, i%col_size]
        label = labels[i]
        pred_label = pred_labels[i]
        prob = probs[i][pred_label]
        
        if class_names is None:
            title = f'{label} ({pred_label} {prob:.2f})'
        else:
            title = f'{class_names[label]} ({class_names[pred_label]} {prob:.2f})'
            
        if label == pred_label:
            ax.set_title(title)
        else:
            ax.set_title(title, color='red')
        
        ax.imshow(img)
        ax.axis('off')
        
    plt.show()