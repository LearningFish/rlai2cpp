/*
#######################################################################
# Copyright (C)                                                       #
# 2018 Donghai He(gsutilml@gmail.com)                                 #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#
# This is a reproduction of the plot shown in Figure 13.1
# in Chapter 13, "Policy Gradient Methods". Book draft May 27, 2018.
#
*/

#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
using namespace std;

// Short corridor environment, see Example 13.1
class Env{
private:
    int s;
public:
    Env() { reset(); }
    void reset() {s = 0;}
    // Args:
    //     go_right (bool): chosen action
    // Returns:
    //     tuple of (reward, episode terminated?)
    pair<int, bool> step(bool go_right){
        if (s == 0 || s == 2){
            s = go_right ? s+1 : max(0, s - 1);
        }
        else{
            //# self.s == 1
            s = go_right ? s-1 : s+1;
        }
        return (s == 3)? 
            //# terminal state
            make_pair(0, true) : make_pair(-1, false);
    }
};

class cutil{
public:
    static vector<double> add(vector<double>& x, vector<double>& y)
    {
        auto r = x;
        for (unsigned int i=0;i<r.size();i++) r[i] += y[i];
        return r;
    }
    static vector<double> exp(vector<double>& x)
    {
        auto r = x;
        for (auto &v : r) v = std::exp(v);
        return r;    
    }
    static vector<double> softmax(vector<double>& x)
    {
        vector<double> t(x.size());
        double maxx = *max_element(x.begin(), x.end());
        double sum = 0;
        for (unsigned int i=0;i<x.size();i++){
            t[i] = std::exp(x[i] - maxx);
            sum += t[i];
        }
        for (auto &x : t) x /= sum;
        return t;
    }   
    // a - 1*1
    // v - 1*n
    static vector<double> dot(double a, vector<double>& v){
        auto r = v;
        for (auto &col : r) col = a * col;
        return r;
    }
    // a - 1*1
    // A - n*m
    static vector<vector<double>> dot(double a, vector<vector<double>>& A){
        auto r = A;
        for (auto &row : r){
            for (auto &col : row)
                col = a * col;
        }
        return r;
    }
    // v - 1*n
    // A - n*m 
    // res - 1*m
    static vector<double> dot(vector<double>& v, vector<vector<double>>& A){
        unsigned int n = A.size(), m=A.front().size(); 
        vector<double> res(m,0);
        for (int col=0;col<m;col++){
            for (int row=0;row<n;row++)
                res[col] += v[row] * A[row][col];
        }
        return res;
    } 
};

// Agent that follows algorithm
// 'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
class Agent{
    public:
        vector<double> theta;
        double alpha;
        double gamma;
        vector<vector<double>> x;
        vector<double> rewards;
        vector<double> actions;
        default_random_engine gen;
    public:
    Agent(double alpha, double gamma){
        //# set values such that initial conditions correspond to left-epsilon greedy
        theta = vector<double>({-1.47, 1.47});
        alpha = alpha;
        gamma = gamma;
        //# first column - left, second - right
        x = vector<vector<double>>({{0, 1},{1, 0}});
    }

    vector<double> get_pi(){
        auto h = cutil::dot(theta, x);
        double maxh = *max_element(h.begin(), h.end());
        vector<double> maxhv(h.size, -maxh);
        auto addv = cutil::add(h, maxhv);
        auto t = cutil::exp(addv);
        double sumt = accumulate(t.begin(), t.end(), 0);
        auto pmf = cutil::dot(1/sumt,t);
        //# never become deterministic, 
        //# guarantees episode finish
        int imin = distance(pmf.begin(), min_element(pmf.begin(), pmf.end()));
        double epsilon = 0.05;

        if (pmf[imin] < epsilon){
            pmf.assign(pmf.size(), 1 - epsilon);
            pmf[imin] = epsilon;
        }
        return pmf;
    }

    double get_p_right(){
        return get_pi()[1];
    }

    bool choose_action(int reward){
        rewards.push_back(reward);
        auto pmf = get_pi();
        binomial_distribution bernouli(pmf[1]);
        bool go_right = bernouli(gen);
        actions.push_back(go_right);
        return go_right;
    }

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            if self.actions[i]:
                j = 1
            else:
                j = 0

            pmf = self.get_pi()
            grad_lnpi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_lnpi
            self.theta += update 
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []
};

def trial(num_episodes, alpha, gamma):
    env = Env()
    agent = Agent(alpha=alpha, gamma=gamma)

    g1 = np.zeros(num_episodes)
    p_right = np.zeros(num_episodes)

    for episode_idx in range(num_episodes):
        # print("Episode {}".format(episode_idx))
        rewards_sum = 0
        reward = None
        env.reset()

        while True:
            go_right = agent.choose_action(reward)
            reward, episode_end = env.step(go_right)
            rewards_sum += reward

            if episode_end:
                agent.episode_end(reward)
                #print('rewards_sum: {}'.format(rewards_sum))
                # decay alpha with time
                #agent.alpha *= 0.995
                break

        g1[episode_idx] = rewards_sum
        p_right[episode_idx] = agent.get_p_right()


    return (g1, p_right)


def run():
    num_trials = 1000
    num_episodes = 1000
    alpha = 2e-4
    gamma = 1

    g1 = np.zeros((num_trials, num_episodes))
    p_right = np.zeros((num_trials, num_episodes))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    res = [pool.apply_async(trial, (num_episodes, alpha, gamma)) for trial_idx in range(num_trials)]

    for trial_idx, r in enumerate(res):
        print("Trial {}".format(trial_idx))
        out = r.get()
        g1[trial_idx, :] = out[0]
        p_right[trial_idx, :] = out[1]

    avg_rewards_sum = np.mean(g1, axis=0)
    avg_p_right = np.mean(p_right, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.arange(num_episodes) + 1, avg_rewards_sum, color="blue")
    ax1.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls="dashed", color="red", label="-11.6")
    ax1.set_ylabel("Value of the first state")
    ax1.set_xlabel("Episode number")
    ax1.set_title("REINFORNCE Monte-Carlo Policy-Gradient Control (episodic) \n"
                 "on a short corridor with switched actions.")
    ax1.legend(loc="lower right")
    # ax1.set_yticks(np.sort(np.append(ax2.get_yticks(), -11.6)))

    ax2.plot(np.arange(num_episodes) + 1, avg_p_right, color="blue")
    ax2.plot(np.arange(num_episodes) + 1, 0.58 * np.ones(num_episodes), ls="dashed", color="red", label="0.58")
    ax2.set_ylabel("Agent's probability of going right")
    ax2.set_xlabel("Episode number\n\n alpha={}. Averaged over {} trials".format(alpha, num_trials))
    ax2.legend(loc="lower right")
    # ax2.set_yticks(np.append(ax2.get_yticks(), 0.58))

    fig.tight_layout()
    plt.show()
    # plt.savefig("out.png")

if __name__ == "__main__":
    run()