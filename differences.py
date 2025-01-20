from matplotlib import pyplot as plt
import numpy as np

def plot():
    data = []
    with open("rankings/ranking_all_boy.txt","r") as f:
        for line in f:
            s = line[:-1].split()
            data.append((s[0],float(s[1]),float(s[2])))

    fsize = 16
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    N = 5
    sort = sorted(data,key=lambda x: x[1],reverse=True)

    probs = [(i[0],i[1] - i[2],i[1],i[2]) for i in sort]
    probs2 = [(i[0],i[2] - i[1],i[1],i[2]) for i in sort]
    probs = sorted(probs,key=lambda x: x[1],reverse=True)
    probs2 = sorted(probs2,key=lambda x: x[1],reverse=True)
    names1 = [i[0] for i in probs][:N]
    names2 = [i[0] for i in probs2][:N]
    probsM = [i[2] for i in probs][:N]
    probsW = [i[3] for i in probs][:N]

    probs2M = [i[2] for i in probs2][:N]
    probs2W = [i[3] for i in probs2][:N]

    width = 0.2
    x = np.arange(len(names1))

    ax.bar(x,probsM,width=width,label="man")
    ax.bar(x+width,probsW,width=width,label="woman")

    ax.set_xticks(x+width,names1)

    ax.set_title("Largest differenec man",fontsize=fsize)
    ax.set_ylabel("Probability",fontsize=fsize)
    ax.set_xlabel("job",fontsize=fsize)
    ax.tick_params(axis="both",labelsize=fsize)
    ax.legend()

    x = np.arange(len(names2))
    ax2.bar(x,probs2M,width=width,label="man")
    ax2.bar(x+width,probs2W,width=width,label="woman")

    ax2.set_xticks(x+width,names2)

    ax2.set_title("Largest difference woman",fontsize=fsize)
    ax2.set_ylabel("Probability",fontsize=fsize)
    ax2.set_xlabel("job",fontsize=fsize)
    ax2.tick_params(axis="both",labelsize=fsize)
    ax2.legend()

    # fig.tight_layout()

    plt.show()



if __name__=="__main__":
    plot()
