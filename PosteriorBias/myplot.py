import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import linear_model
import numpy as np

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

