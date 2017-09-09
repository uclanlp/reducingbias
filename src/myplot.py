import matplotlib.pyplot as plt
import pandas as pd

    #################################For imSitu #############################
def plot_tr_ax_gender(res, margin, is_dev):
    if is_dev == 1:
        fn = "dev"
    else:
        fn = "test"
    plt.xlabel("training gender ratio")
    plt.ylabel("predicted gender ratio  ")
#     plt.title("Gender ratio predicted in " + fn +" with margin "+(str(margin)))

    plt.plot(res['training_ratio'], res['bef_ratio'], 'r.')
    plt.plot(res[(res['bef_ratio'] <= res['training_ratio'] + margin) &
                 (res['bef_ratio'] >= res['training_ratio'] - margin)]['training_ratio'],
             res[(res['bef_ratio'] <= res['training_ratio'] + margin) &
                (res['bef_ratio'] >= res['training_ratio'] - margin)]['bef_ratio'], 'g.')
    plt.plot(res['training_ratio'], res['training_ratio'], 'b-')
    plt.plot(res['training_ratio'], res['training_ratio'] + margin, 'b--')
    plt.plot(res['training_ratio'], res['training_ratio'] - margin, 'b--')
    plt.show()
    plt.close()


    plt.xlabel("training gender ratio")
    plt.ylabel("predicted gender ratio  ")
#     plt.title("Gender ratio predicted in " + fn +" with margin "+(str(margin)))

    plt.plot(res['training_ratio'], res['after_ratio'], 'r.')
    plt.plot(res[(res['after_ratio'] <= res['training_ratio'] + margin) &
                 (res['after_ratio'] >= res['training_ratio'] - margin)]['training_ratio'],
             res[(res['after_ratio'] <= res['training_ratio'] + margin) &
                (res['after_ratio'] >= res['training_ratio'] - margin)]['after_ratio'], 'g.')
    plt.plot(res['training_ratio'], res['training_ratio'], 'b-')
    plt.plot(res['training_ratio'], res['training_ratio'] + margin, 'b--')
    plt.plot(res['training_ratio'], res['training_ratio'] - margin, 'b--')
    plt.show()
    plt.close()


def plot_mean_ratio_dis_trax_gender(res,margin, is_dev):
    tmp_bef = []
    tmp_aft = []
    tmp_tra = []
    end = 0
    res['bef_diff_up_margin'] = 0
    res['bef_diff_lo_margin'] = 0
    res['after_diff_up_margin'] = 0
    res['after_diff_lo_margin'] = 0

    res['bef_diff_up_margin'][res['bef_ratio'] > res['training_ratio'] + margin] = res['bef_ratio'] - (res['training_ratio'] + margin)
    res['bef_diff_lo_margin'][res['bef_ratio'] < res['training_ratio'] - margin] = res['bef_ratio'] - (res['training_ratio'] - margin)
    res['after_diff_up_margin'][res['after_ratio'] > res['training_ratio'] + margin] = res['after_ratio'] - (res['training_ratio'] + margin)
    res['after_diff_lo_margin'][res['after_ratio'] < res['training_ratio'] - margin] = res['after_ratio'] - (res['training_ratio'] - margin)

    if is_dev == 1:
        fn = "dev"
    else:
        fn = "test"
    for i in range(res.shape[0]):
        end = i + 20
        if end < res.shape[0]:
            tmp_bef.append((res['training_ratio'].iloc[i+9],
                            (abs(res['bef_diff_up_margin'].iloc[i:end]).sum() +
                             abs(res['bef_diff_lo_margin'].iloc[i:end]).sum()) / 20) )
            tmp_aft.append((res['training_ratio'].iloc[i+9],
                            (abs(res['after_diff_up_margin'].iloc[i:end]).sum() +
                             abs(res['after_diff_lo_margin'].iloc[i:end]).sum()) / 20))
    bef_mean_dis = pd.DataFrame(tmp_bef, columns = ['training_ratio','bef_mean_diff'])
    aft_mean_dis = pd.DataFrame(tmp_aft, columns = ['training_ratio','aft_mean_diff'])
    plt.plot(bef_mean_dis.training_ratio, bef_mean_dis['bef_mean_diff'], 'r--')
    plt.plot(aft_mean_dis.training_ratio, aft_mean_dis['aft_mean_diff'], 'b-')
    plt.xlabel("training gender ratio")
    plt.ylabel("mean bias amplified")
    plt.show()

###############used for coco dataset    ############
def coco_plot_train_x(df, margin):
    plt.plot(df['train_ratio'], df['train_ratio'], 'b-')
    plt.plot(df['train_ratio'], df['train_ratio'] + margin, 'b--')
    plt.plot(df['train_ratio'], df['train_ratio'] - margin, 'b--')
    plt.plot(df['train_ratio'], df['bef_ratio'], 'g.')
    plt.plot(df[abs(df['bef_ratio'] - df['train_ratio']) >  margin]['train_ratio'],
             df[abs(df['bef_ratio'] - df['train_ratio']) >  margin]['bef_ratio'], 'r.')
    plt.xlabel('training gender ratio')
    plt.ylabel('predicted gender ratio ')
    plt.show()

    plt.plot( df['train_ratio'], df['train_ratio'], 'b-')
    plt.plot(df['train_ratio'], df['train_ratio'] + margin, 'b--')
    plt.plot(df['train_ratio'], df['train_ratio'] - margin, 'b--')
    plt.plot(df['train_ratio'], df['aft_ratio'], 'g.')
    plt.plot(df[abs(df['aft_ratio'] - df['train_ratio']) >  margin]['train_ratio'],
             df[abs(df['aft_ratio'] - df['train_ratio']) >  margin]['aft_ratio'], 'r.')
    plt.xlabel('training gender ratio')
    plt.ylabel('predicted gender ratio ')
    plt.show()


def coco_plot_mean_ratio_dis_trax(res, margin, is_dev = 1):
    tmp_bef = []
    tmp_aft = []
    tmp_tra = []
    end = 0
    res['ori_diff_up_margin'] = 0
    res['ori_diff_lo_margin'] = 0
    res['after_diff_up_margin'] = 0
    res['after_diff_lo_margin'] = 0

    res['ori_diff_up_margin'][res['bef_ratio'] > res['train_ratio'] + margin] = res['bef_ratio'] - (res['train_ratio'] + margin)
    res['ori_diff_lo_margin'][res['bef_ratio'] < res['train_ratio'] - margin] = res['bef_ratio'] - (res['train_ratio'] - margin)
    res['after_diff_up_margin'][res['aft_ratio'] > res['train_ratio'] + margin] = res['aft_ratio'] - (res['train_ratio'] + margin)
    res['after_diff_lo_margin'][res['aft_ratio'] < res['train_ratio'] - margin] = res['aft_ratio'] - (res['train_ratio'] - margin)
    if is_dev == 1:
        fn = "dev"
    else:
        fn = "test"
    for i in range(res.shape[0]):
        end = i + 5
        if end < res.shape[0]:
            tmp_bef.append((res['train_ratio'].iloc[i+2],
                            (abs(res['ori_diff_up_margin'].iloc[i:end]).sum() +
                             abs(res['ori_diff_lo_margin'].iloc[i:end]).sum()) / 5) )
            tmp_aft.append((res['train_ratio'].iloc[i+2],
                            (abs(res['after_diff_up_margin'].iloc[i:end]).sum() +
                             abs(res['after_diff_lo_margin'].iloc[i:end]).sum()) / 5))
    bef_mean_dis = pd.DataFrame(tmp_bef, columns = ['train_ratio','bef_mean_diff'])
    aft_mean_dis = pd.DataFrame(tmp_aft, columns = ['train_ratio','aft_mean_diff'])
    plt.plot(bef_mean_dis.train_ratio, bef_mean_dis['bef_mean_diff'], 'r--')
    plt.plot(aft_mean_dis.train_ratio, aft_mean_dis['aft_mean_diff'], 'b-')
    plt.xlabel("training gender ratio")
    plt.ylabel("mean bias amplification")
    plt.show()




#############for vSRL example #########
def plot_bias(res, vSRL, is_dev):
    if is_dev == 1:
        fn = "dev"
    else:
        fn = "test"
    plt.xlabel("training gender ratio")
    plt.ylabel("predicted gender ratio  ")
    plt.plot(res['training_ratio'], res['bef_ratio'], 'r.')
    plt.plot(res['training_ratio'], res['training_ratio'], 'b-')
    if vSRL == 1:
        inter_words = ['washing','shopping', 'driving', 'coaching']
        for word in inter_words:
            plt.plot(res[res['verb'] == word]['training_ratio'].values[0], res[res['verb'] == word]['bef_ratio'].values[0], 'k*')
            plt.annotate(word , 
                         xy=(res[res['verb'] == word]['training_ratio'].values[0], res[res['verb'] == word]['bef_ratio'].values[0]), 
                         xytext=(res[res['verb'] == word]['training_ratio'].values[0] + 0.04, res[res['verb'] == word]['bef_ratio'].values[0]), color = 'k')
    else:
        inter_words = ['knife', 'fork', 'snowboard', 'boat']
        for word in inter_words:
            plt.plot(res[res['object'] == word]['training_ratio'].values[0], res[res['object'] == word]['bef_ratio'].values[0], 'k*')
            plt.annotate(word , 
                         xy=(res[res['object'] == word]['training_ratio'].values[0], res[res['object'] == word]['bef_ratio'].values[0]), 
                         xytext=(res[res['object'] == word]['training_ratio'].values[0] + 0.02, res[res['object'] == word]['bef_ratio'].values[0]), color = 'k')

    plt.show()
    plt.close()
