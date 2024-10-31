from lifelines import KaplanMeierFitter
import numpy as np


def get_integrated(x, times):
    return np.trapz(x,times) / (max(times)-min(times))

# IPCW brier
def brier(preds,events,times,train_e, train_t,LT,pred_times):
    kmf_c = KaplanMeierFitter()
    # not e to fit for censoring!
    kmf_c.fit(train_t,~train_e)
    batch_size,length = len(events),len(pred_times)
    w1 = np.zeros([batch_size,length],dtype=np.float32)
    w2 = np.zeros([batch_size,length],dtype=np.float32)

    G1 = np.zeros([batch_size,length],dtype=np.float32)
    G2 = np.zeros([batch_size,length],dtype=np.float32)
    event_mat = np.repeat(events.reshape(-1,1),repeats=length,axis=1)
    times_mat = np.repeat(times.reshape(-1,1),repeats=length,axis=1)
    pred_times_mat = np.repeat(pred_times.reshape(1,-1),repeats=batch_size,axis=0)

    LT_factor = kmf_c.predict(LT)

    w1[(times_mat<=pred_times_mat) & (event_mat)] = 1
    # to approach from - :
    G1= np.repeat((kmf_c.predict(times-1e-9).to_numpy() / LT_factor).reshape(-1,1),repeats=length,axis=1)
    w2[times_mat>pred_times_mat] = 1
    G2 = np.repeat((kmf_c.predict(pred_times).to_numpy() / LT_factor).reshape(1,-1),repeats=batch_size,axis=0)

    B_score = np.mean((w1*preds**2/G1 + w2*(1-preds)**2/G2),axis=0)
    return B_score


# compute brier across seeds faster by passing in fitted kmf_c
def brier_fast(preds,events,times,kmf_c,LT,pred_times):
    batch_size,length = len(events),len(pred_times)
    w1 = np.zeros([batch_size,length],dtype=np.float32)
    w2 = np.zeros([batch_size,length],dtype=np.float32)

    G1 = np.zeros([batch_size,length],dtype=np.float32)
    G2 = np.zeros([batch_size,length],dtype=np.float32)
    event_mat = np.repeat(events.reshape(-1,1),repeats=length,axis=1)
    times_mat = np.repeat(times.reshape(-1,1),repeats=length,axis=1)
    pred_times_mat = np.repeat(pred_times.reshape(1,-1),repeats=batch_size,axis=0)

    LT_factor = kmf_c.predict(LT)

    w1[(times_mat<=pred_times_mat) & (event_mat)] = 1
    # to approach from - :
    G1= np.repeat((kmf_c.predict(times-1e-9).to_numpy() / LT_factor).reshape(-1,1),repeats=length,axis=1)
    w2[times_mat>pred_times_mat] = 1
    G2 = np.repeat((kmf_c.predict(pred_times).to_numpy() / LT_factor).reshape(1,-1),repeats=batch_size,axis=0)

    B_score = np.mean((w1*preds**2/G1 + w2*(1-preds)**2/G2),axis=0)
    return B_score