import numpy as np
import os
import time
import librosa

np.random.seed(int(time.time()))

len_of_one_cut = 16000
max_ratio = 320
ratio_cut_and_step = 4
assert max_ratio >= ratio_cut_and_step
assert len_of_one_cut % ratio_cut_and_step == 0
step = len_of_one_cut // ratio_cut_and_step
cut_of_each_utter = ratio_cut_and_step + 1
len_of_one_utter = len_of_one_cut + (cut_of_each_utter - 1) * step
W = 127
H = 128
C = 1

trainpath = '/data/audio/timit/train'

def Utter2MultiFrame(utter, len_of_cut, step, num_of_cut):
    # print(len(utter))
    ret = []
    for i in range(num_of_cut):
        start = i * step
        cut = utter[start: start + len_of_cut]
        assert len(cut) == len_of_cut
        ret.append(cut)
    return np.array(ret)


# a test to read file

class Data_Read_Timit(object):
    def __init__(self, sumpath, outset_each_divide=False, ratio_pn=199, utter_once_batch=1,
                 len_of_utter=len_of_one_utter, len_of_cut=len_of_one_cut, step=step, cut_of_utter=cut_of_each_utter):
        self.sumpath = sumpath
        self.outset_each_divide = outset_each_divide
        self.ratio_pn = ratio_pn
        self.labels_oncebatch = self.ratio_pn + 1  # how many types of label would be chosen each batch
        self.utter_once_batch = utter_once_batch
        self.batch_size = self.labels_oncebatch * self.utter_once_batch

        self.len_of_utter = len_of_utter
        self.len_of_cut = len_of_cut
        self.step = step
        self.cut_of_utter = cut_of_utter

        self.InSet_Train_Data = []
        self.InSet_Train_Label = []
        self.InSet_Num = 1

        self.len_train = None

    def ReadDataAndConvertIntoSpectrogram(self):
        print("start to read audio and convert it into num*16000")
        count_label = 0
        addstep = len_of_one_utter // 2
        for p1 in os.listdir(self.sumpath):   # district
            p11 = os.path.join(self.sumpath, p1)
            for p2 in os.listdir(p11):        # people
                p21 = os.path.join(p11, p2)
                addnum = 0
                for p3 in os.listdir(p21):    # people's audio
                    if p3[-3:] == 'wav':
                        p4 = os.path.join(p21, p3)
                        audio, fs = librosa.load(p4, sr=None)
                        al = len(audio)
                        start = 0
                        while start + len_of_one_utter <= al:
                            tmpx = audio[start : start + len_of_one_utter]
                            start += addstep
                            addnum += 1
                            tmpd = Utter2MultiFrame(tmpx, len_of_one_cut, step, cut_of_each_utter)
                            self.InSet_Train_Data.append(tmpd)
                            self.InSet_Train_Label.append(count_label)
                if addnum > 0:
                    print(p21, addnum)
                    count_label += 1
                    #if count_label > 10:
                    #    break
            #if count_label > 10:
            #    break
        self.InSet_Num = len(set(self.InSet_Train_Label))
        print("data read over")
        self.InSet_Train_Data = np.array(self.InSet_Train_Data)
        self.InSet_Train_Label = np.array(self.InSet_Train_Label)
        self.len_train = len(self.InSet_Train_Data)
        # num_sample cut_of_each 16000
        print("data made over")

    def TripletDataGeter(self, label_inset):
        anchor_ret = self.InSet_Train_Data[self.InSet_Train_Label == label_inset]
        dl = len(anchor_ret)
        Others = self.InSet_Train_Data[self.InSet_Train_Label != label_inset]
        slt = np.random.choice(len(Others), dl)
        neg_ret = Others[slt]
        neg_l = self.InSet_Train_Label[self.InSet_Train_Label != label_inset][slt]
        sftidxs = np.arange(dl)
        np.random.shuffle(sftidxs)
        pos_ret = anchor_ret[sftidxs]
        ret_apl = np.ones(dl) * label_inset
        return anchor_ret, pos_ret, neg_ret, ret_apl, neg_l

    def RandonOneBatchData_Train(self):
        randomIdx = np.random.choice(self.len_train, self.batch_size)
        sltdata = self.InSet_Train_Data[randomIdx]
        sltlabel = self.InSet_Train_Label[randomIdx]
        return sltdata, sltlabel

    def DataBatch(self, Data, Label, Order, remain=0):
        if remain > 0:
            retdata = Data[-remain:]
            retlabel = Label[-remain:]

        else:
            retdata = Data[Order * self.batch_size: (Order + 1) * self.batch_size]
            retlabel = Label[Order * self.batch_size: (Order + 1) * self.batch_size]

        return retdata, retlabel


class FeatureVectors(object):
    def __init__(self, tifv=0, til=0, eifv=0, eil=0, eofv=0, eol=0, sfv=0, sl=0, ni=0, batch_size=200):
        # part of data
        self.train_inset_feature_vectors = tifv
        self.train_inset_labels = til
        self.test_inset_feature_vectors = eifv
        self.test_inset_labels = eil
        self.test_outdet_feature_vectors = eofv
        self.test_outset_labels = eol
        self.standard_feature_vectors = sfv
        self.standard_labels = sl
        # part of parameter
        self.num_inset = ni
        self.batch_size = batch_size
        print("feature vector data set made over")

    def SeparateByLabel(self, data, label):
        newdata = []
        for i in range(self.num_inset):
            tmp_i = data[label == i]
            newdata.append(tmp_i)
        return np.array(newdata)

    def BatchData(self, data, label, rounds, remain=0):
        if remain > 0:
            return data[-remain:], label[-remain:]
        else:
            return data[rounds * self.batch_size: (rounds + 1) * self.batch_size], label[rounds * self.batch_size: (
                                                                                                                   rounds + 1) * self.batch_size]

