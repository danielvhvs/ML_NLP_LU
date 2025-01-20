from matplotlib import pyplot as plt
import numpy as np

def plot():
    data = []
    with open("rankings/ranking_all_as.txt","r") as f:
        for line in f:
            s = line[:-1].split()
            data.append((s[0],float(s[1]),float(s[2])))

    fsize = 16
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    N = 5
    sort = sorted(data,key=lambda x: x[1],reverse=True)
    names = [i[0] for i in sort][:N]
    probs = np.array([i[1] for i in sort])[:N]/40
    probs2 = np.array([i[2] for i in sort])[:N]/40
    width = 0.2
    x = np.arange(len(names))

    ax.bar(x,probs,width=width,label="man")
    ax.bar(x+width,probs2,width=width,label="woman")

    ax.set_xticks(x+width,names)

    ax.set_title("Most common job predictions man all sentences",fontsize=fsize)
    ax.set_ylabel("Probability",fontsize=fsize)
    ax.set_xlabel("job",fontsize=fsize)
    ax.tick_params(axis="both",labelsize=fsize)
    ax.legend()


    sort = sorted(data,key=lambda x: x[2],reverse=True)
    names = [i[0] for i in sort][:N]
    probs = np.array([i[1] for i in sort])[:N]/40
    probs2 = np.array([i[2] for i in sort])[:N]/40
    ax2.bar(x,probs,width=width,label="man")
    ax2.bar(x+width,probs2,width=width,label="woman")

    ax2.set_xticks(x+width,names)

    ax2.set_title("Most common job predictions woman all sentences",fontsize=fsize)
    ax2.set_ylabel("Probability",fontsize=fsize)
    ax2.set_xlabel("job",fontsize=fsize)
    ax2.tick_params(axis="both",labelsize=fsize)
    ax2.legend()

    # fig.tight_layout()

    plt.show()



if __name__=="__main__":
    plot()
