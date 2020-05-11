import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import linear_model
import numpy as np

    ################################# For imSitu #############################
def plot_tr_ax_gender(res, margin, filename):
    # figure for inference results (before and after)
    plt.tick_params(labelsize = 18)
    plt.xlabel("bias in training set", fontsize = 25)
    plt.ylabel("bias in predictions", fontsize = 25)
    # plt.title("Inference result with margin "+(str(margin))+"_before", fontsize = 20)
    # plt.axes([0,1,0,1])

    plt.plot(res['training_ratio'], res['bef_ratio'], 'r.')
    plt.plot(res[(res['bef_ratio'] <= res['training_ratio'] + margin) &
                (res['bef_ratio'] >= res['training_ratio'] - margin)]['training_ratio'],
                res[(res['bef_ratio'] <= res['training_ratio'] + margin) &
                (res['bef_ratio'] >= res['training_ratio'] - margin)]['bef_ratio'], 'g.')
    # plt.legend(["verb violating constraints","verb satisfying constraints"], fontsize = 14)
    plt.plot(res['training_ratio'], res['training_ratio'], color = 'b', linestyle = '-')
    plt.plot(res['training_ratio'], res['training_ratio'] + margin, color = 'b', linestyle = '--')
    plt.plot(res['training_ratio'], res['training_ratio'] - margin, color = 'b', linestyle = '--')

    reg = linear_model.LinearRegression()
    res_y = res['bef_ratio'].values
    res_x = res['training_ratio'].values
    reg.fit (res_x.reshape(-1,1), res_y)
    x = np.linspace(0.1, 0.88, num=10)
    y = reg.predict(x.reshape(-1, 1))
    plt.plot(x, y, linestyle='-', color='black', linewidth=2)
    plt.tight_layout()
    print ("Inference before: k={}, b={}".format(reg.coef_, reg.intercept_))
    
    plt.savefig("./results/"+filename+"Inference result with margin "+(str(margin))+"_before.pdf")
    plt.show()
    plt.close()


    plt.tick_params(labelsize = 18)
    plt.xlabel("bias in training set", fontsize = 25)
    plt.ylabel("bias in predictions", fontsize = 25)
    # plt.title("Inference result with margin "+(str(margin))+"_after", fontsize = 20)
    
    plt.plot(res['training_ratio'], res['after_ratio'], 'r.')
    plt.plot(res[(res['after_ratio'] <= res['training_ratio'] + margin) &
                (res['after_ratio'] >= res['training_ratio'] - margin)]['training_ratio'],
                res[(res['after_ratio'] <= res['training_ratio'] + margin) &
                (res['after_ratio'] >= res['training_ratio'] - margin)]['after_ratio'], 'g.')
    plt.plot(res['training_ratio'], res['training_ratio'], 'b-')
    plt.plot(res['training_ratio'], res['training_ratio'] + margin, 'b--')
    plt.plot(res['training_ratio'], res['training_ratio'] - margin, 'b--')

    reg = linear_model.LinearRegression()
    res_y = res['after_ratio'].values
    res_x = res['training_ratio'].values
    reg.fit (res_x.reshape(-1,1), res_y)
    x = np.linspace(0.1, 0.88, num=10)
    y = reg.predict(x.reshape(-1, 1))
    plt.plot(x, y, linestyle='-', color='black', linewidth=2)
    plt.tight_layout()
    print ("Inference after: k={}, b={}".format(reg.coef_, reg.intercept_))

    plt.savefig("./results/"+filename+"Inference result with margin "+(str(margin))+"_after.pdf")
    plt.show()
    plt.close()


    # figure for posterior ratio (before and after)
    plt.tick_params(labelsize = 18)
    plt.xlabel("bias in training set", fontsize = 25)
    plt.ylabel("bias in predictions", fontsize = 25)
    # plt.title("Posterior distribution with margin "+(str(margin))+"_before", fontsize = 20)
    
    plt.plot(res['training_ratio'], res['bef_ratio_PR'], 'r.')
    plt.plot(res[(res['bef_ratio_PR'] <= res['training_ratio'] + margin) &
                (res['bef_ratio_PR'] >= res['training_ratio'] - margin)]['training_ratio'],
                res[(res['bef_ratio_PR'] <= res['training_ratio'] + margin) &
                (res['bef_ratio_PR'] >= res['training_ratio'] - margin)]['bef_ratio_PR'], 'g.')
    plt.plot(res['training_ratio'], res['training_ratio'], 'b-')
    plt.plot(res['training_ratio'], res['training_ratio'] + margin, 'b--')
    plt.plot(res['training_ratio'], res['training_ratio'] - margin, 'b--')

    reg = linear_model.LinearRegression()
    res_y = res['bef_ratio_PR'].values
    res_x = res['training_ratio'].values
    reg.fit (res_x.reshape(-1,1), res_y)
    x = np.linspace(0.1, 0.88, num=10)
    y = reg.predict(x.reshape(-1, 1))
    plt.plot(x, y, linestyle='-', color='black', linewidth=2)
    plt.tight_layout()
    print ("Posterior after: k={}, b={}".format(reg.coef_, reg.intercept_))

    plt.savefig("./results/"+filename+"Posterior distribution with margin "+(str(margin))+"_before.pdf")
    plt.show()
    plt.close()


    plt.tick_params(labelsize = 18)
    plt.xlabel("bias in training set", fontsize = 25)
    plt.ylabel("bias in predictions", fontsize = 25)
    # plt.title("Posterior distribution with margin "+(str(margin))+"_after", fontsize = 20)
    
    plt.plot(res['training_ratio'], res['after_ratio_PR'], 'r.')
    plt.plot(res[(res['after_ratio_PR'] <= res['training_ratio'] + margin) &
                (res['after_ratio_PR'] >= res['training_ratio'] - margin)]['training_ratio'],
                res[(res['after_ratio_PR'] <= res['training_ratio'] + margin) &
                (res['after_ratio_PR'] >= res['training_ratio'] - margin)]['after_ratio_PR'], 'g.')
    plt.plot(res['training_ratio'], res['training_ratio'], 'b-')
    plt.plot(res['training_ratio'], res['training_ratio'] + margin, 'b--')
    plt.plot(res['training_ratio'], res['training_ratio'] - margin, 'b--')

    reg = linear_model.LinearRegression()
    res_y = res['after_ratio_PR'].values
    res_x = res['training_ratio'].values
    reg.fit (res_x.reshape(-1,1), res_y)
    x = np.linspace(0.1, 0.88, num=10)
    y = reg.predict(x.reshape(-1, 1))
    plt.plot(x, y, linestyle='-', color='black', linewidth=2)
    print ("Posterior after: k={}, b={}".format(reg.coef_, reg.intercept_))
    plt.tight_layout()

    plt.savefig("./results/"+filename+"Posterior distribution with margin "+(str(margin))+"_after.pdf")
    plt.show()
    plt.close()


def get_violated_verbs(res, margin):
    print ("[Inference]: \nBefore calibrating, %d verbs are not satisfied"%(res[abs(res['bef_diff']) > margin].count().verb_id))
    print ("After calibrating, %d verbs are not satisfied"%(res[abs(res['after_diff']) > margin].count().verb_id))
    print ("[Posterior probability]: \nBefore calibrating, %d verbs are not satisfied"%(res[abs(res['bef_diff_PR']) > margin].count().verb_id))
    print ("After calibrating, %d verbs are not satisfied"%(res[abs(res['after_diff_PR']) > margin].count().verb_id))
    return (res[abs(res['bef_diff']) > margin].count().verb_id, res[abs(res['after_diff']) > margin].count().verb_id, 
            res[abs(res['bef_diff_PR']) > margin].count().verb_id, res[abs(res['after_diff_PR']) > margin].count().verb_id)


def get_bias_score(res, margin):
    # calculate the bias score of inference results
    print ("[Inference results]")
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['bef_ratio']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['bef_ratio'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("Before calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))
    #calculate the bias score after LR
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['after_ratio']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['after_ratio'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("After calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))

    # calculate the bias score of distribution
    print ("[Posterior distribution]")
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['bef_ratio_PR']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['bef_ratio_PR'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("Before calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))
    #calculate the bias score after LR
    woman_bias = (res[res['training_ratio'] < 0.5]['training_ratio'] - res[res['training_ratio'] < 0.5]['after_ratio_PR']).sum()
    man_bias = (res[res['training_ratio'] > 0.5]['after_ratio_PR'] - res[res['training_ratio'] > 0.5]['training_ratio']).sum()
    print ("After calibrating, mean bias score for all verbs = (woman_bias + man_bias)/#(verbs): %s "%(str((woman_bias + man_bias)/res.shape[0])))


if __name__=='__main__':
    with open("./results/results_lr0-01_margin0-001") as f:
        res = f.readlines()

    cons_verbs = []
    training_ratio = {}
    ori_ratio = {}
    after_ratio = {}
    ori_ratio_PR = {}
    after_ratio_PR = {}
    for res_ in res:
        results = res_.split('\t')
        verb_id = int(results[0].split(' ')[0])
        verb = results[0].split(' ')[1]
        cons_verbs.append(verb_id)
        training_ratio[verb_id] = float(results[1].split(':')[1])
        ori_ratio_PR[verb_id] = float(results[2].split(':')[1])
        after_ratio_PR[verb_id] = float(results[3].split(':')[1])
        ori_ratio[verb_id] = float(results[4].split(':')[1])
        after_ratio[verb_id] = float(results[5].split(':')[1])

    golden_verbs_df = pd.DataFrame([verb for verb in cons_verbs], columns = ['verb_id'])
    train_df = pd.DataFrame(training_ratio.items(), columns = ["verb_id", "training_ratio"])
    ori_df = pd.DataFrame(ori_ratio.items(), columns=['verb_id', 'bef_ratio'])
    after_df = pd.DataFrame(after_ratio.items(), columns=['verb_id', 'after_ratio'])
    ori_df_PR = pd.DataFrame(ori_ratio_PR.items(), columns=['verb_id', 'bef_ratio_PR'])
    after_df_PR = pd.DataFrame(after_ratio_PR.items(), columns=['verb_id', 'after_ratio_PR'])
    tmp = ori_df.merge(after_df).merge(ori_df_PR).merge(after_df_PR)
    com_df = train_df.merge(tmp)
    res0 = com_df.sort_values(by = ['training_ratio'], ascending=1)
    res0['bef_diff'] = res0['training_ratio'] - res0['bef_ratio']   # training_ratio - bef_ratio
    res0['after_diff'] = res0['training_ratio'] - res0['after_ratio']   # training_ratio - after_ratio
    res0['bef_diff_PR'] = res0['training_ratio'] - res0['bef_ratio_PR']   # training_ratio - bef_ratio
    res0['after_diff_PR'] = res0['training_ratio'] - res0['after_ratio_PR']   # training_ratio - after_ratio
    resx = res0.merge(golden_verbs_df)
    resx.sort_values(by = ['training_ratio'], ascending=1, inplace = True)
    # del resx['verb_id']
    resx.reset_index(inplace=True, drop = True)
    # print (resx)

    plot_tr_ax_gender(resx, 0.05, "Margin0-001_")
    get_violated_verbs(resx, 0.05)
    get_bias_score(resx, 0.05)
