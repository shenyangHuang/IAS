import keras
import numpy as np
 
class class_accuracy(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self,epoch, logs={}):
        return 

    def on_epoch_end(self, epoch, logs={}):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        num_classes = 0
        for onehot in val_y:
            label = np.argmax(onehot)
            if(label > (num_classes-1)):
                num_classes = label + 1

        class_x=[]
        class_y=[]
        for i in range(0,num_classes):
            class_x.append([])
        for i in range(0,num_classes):
            class_y.append([])

        for i in range(len(val_y)):
            label = np.argmax(val_y[i])
            class_x[label].append(val_x[i])
            class_y[label].append(val_y[i])

        print ("epoch number " + str(epoch))
        for i in range(num_classes):
            c_predict = self.model.predict(np.asarray(class_x[i]))
            correct = np.argmax(class_y[i][0])
            max_num = len(class_y[i])
            correct_num = 0
            for j in range(len(c_predict)):
                predicted = np.argmax(c_predict[j])
                if(predicted == correct):
                    correct_num = correct_num+1
            print ("class " + str(i) + " : " + str(correct_num) + " / " + str(max_num))
            accuracy = float(correct_num) / float(max_num)
            print ("validation accuracy is " + str(accuracy) )

        predict = self.model.predict(self.validation_data[0])
        total = len(self.validation_data[1])
        score = 0
        for i in range(len(self.validation_data[1])):
            p_value = np.argmax(predict[i])
            correct = np.argmax(self.validation_data[1][i])
            if(p_value == correct):
                score = score + 1
        print ("total  : " + str(score) + " / " + str(total))
        accuracy = float(score) / float(total)
        print ("average incremental validation accuracy is " + str(accuracy) )






        





        print ()
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return