#include <cmath>
#include <random>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>
using namespace std;

class Bandit
{
private:
    vector<double> action_prob_;
    default_random_engine rnd_engine;
public:
    int k_;
    double step_size_;
    bool sample_averages_;
    double time_step_;
    double* ucb_param_ = NULL;
    bool gradient_;
    bool gradient_baseline_;
    double average_reward_;
    double true_reward_;
    vector<double> q_true_;   // real reward for each action
    vector<double> q_est_;    // estimated reward for each action
    vector<int> action_count_; // count of chosen for each action
    double epsilon_;
    int best_action_;
public:
    Bandit(int kArm=10, double epsilon=0., double initial=0., double stepSize=0.1, double sampleAverages=false, double* ucb_param=NULL,
                 bool gradient=false, bool gradientBaseline=false, double trueReward=0.)
                 : k_(kArm), epsilon_(epsilon), step_size_(stepSize), sample_averages_(sampleAverages), ucb_param_(ucb_param),
                 gradient_(gradient), gradient_baseline_(gradientBaseline), true_reward_(trueReward)
                 {
                     time_step_ = 0;
                     q_est_.resize(k_,0);
                     normal_distribution<double> norm_dist(0,1);
                     for (int i=0;i<k_;i++){
                         q_true_.push_back(norm_dist(rnd_engine)+true_reward_);
                         q_est_[i] = initial;
                         action_count_.push_back(0);
                     }
                     best_action_ = distance(q_true_.begin(), max_element(q_true_.begin(),q_true_.end()));
                 }
    int action()
    {
        uniform_int_distribution<int> unifn_dist(0,k_-1);
        uniform_real_distribution<double> unifr_dist(0,1);
        if (epsilon_>0 && unifr_dist(rnd_engine)<epsilon_) return unifn_dist(rnd_engine);

        if (ucb_param_!=NULL){
            auto ucb_est = q_est_;
            for (int i=0;i<k_;i++) ucb_est[i] += (*ucb_param_) * sqrt(log(time_step_+1)/(action_count_[i]+1));
        }

        if (gradient_){
            auto exp_est = q_est_;
            for (int i=0;i<k_;i++) exp_est[i]=exp(q_est_[i]);
            auto tot = accumulate(exp_est.begin(), exp_est.end(),0); 
            action_prob_ = move(exp_est);
            for (auto &x : action_prob_) x = x/tot;
            discrete_distribution<int> disc_dist(action_prob_.begin(), action_prob_.end());
            return disc_dist(rnd_engine);
        }
    }

    double sample(int action_index)
    {
        normal_distribution<double> norm_dist(0,1);
        double reward = norm_dist(rnd_engine) + q_true_[action_index];
        time_step_++;
        average_reward_ = (time_step_ - 1.0)/time_step_*average_reward_+reward/time_step_;
        action_count_[action_index] +=1;

        if (sample_averages_){
            q_est_[action_index] += 1.0 / action_count_[action_index] * (reward - q_est_[action_index]);
        }else if (gradient_){
            vector<int> is_action(k_,0);
            is_action[action_index] = 1;
            auto baseline = gradient_baseline_?average_reward_:0;
            for (int i=0;i<k_;i++) q_est_[i] += step_size_ * (reward - baseline) * (is_action[i] - action_prob_[i]);
        }else{
            // constant step size
            q_est_[action_index] += step_size_ * (reward - q_est_[action_index]);
        }
        return reward;
    }
};

void figure2_1()
{
    default_random_engine rnd;
    //normal_distribution<double> norm_dist_main(200,10), norm_dist_extra(0,10);
}

pair<vector<vector<double>>, vector<vector<double>> > bandit_simulation(int num_bandits, int time_step, vector<vector<Bandit*>>& bandits)
{
    vector<vector<double>> best_action_counts(bandits.size(), vector<double>(time_step,0.0));
    vector<vector<double>> average_rewards(bandits.size(), vector<double>(time_step,0.0));
    for (int k=0;k<bandits.size();k++){
        for (int i=0;i<num_bandits; i++){
            for (int t=0;t<time_step; t++){
                auto action_index = bandits[k][i]->action();
                auto reward = bandits[k][i]->sample(action_index);
                average_rewards[k][t] += reward;
                if (action_index==bandits[k][i]->best_action_) best_action_counts[k][t] += 1;
            }
        }
        for (auto &x : best_action_counts[k]) x = x / num_bandits;
        for (auto &x : average_rewards[k]) x = x / num_bandits;
    }
    return make_pair(best_action_counts, average_rewards);
}

// figure 2.2
void epsilon_greedy(int num_bandits, int time_step)
{
    vector<double> epsilons = {0, 0.1, 0.01};
    vector<vector<Bandit*>> bandits;
    for (int i=0;i<epsilons.size();i++){
        vector<Bandit*> vec_bandits;
        for (int j=0;j<num_bandits;j++){
            vec_bandits.push_back(new Bandit());
            vec_bandits.back()->epsilon_ = epsilons[i];
            vec_bandits.back()->sample_averages_ = true;
        }
        bandits.push_back(vec_bandits); 
    }
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    for (auto &v : bandits)
        for (auto p : v)
            delete p;
}

// figure 2.3
void optimistic_initial_values(int num_bandits, int time_step)
{
    vector<vector<Bandit*>> bandits;
    vector<Bandit*> vec_bandits;
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0., 5, 0.1));
    }
    bandits.push_back(vec_bandits); 
    vec_bandits.clear();
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0.1, 0, 0.1));
    }
    bandits.push_back(vec_bandits); 
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    for (auto &v : bandits)
        for (auto p : v)
            delete p;
}

// figure 2.4
void ucb(int num_bandits, int time_step)
{
    vector<vector<Bandit*>> bandits;
    vector<Bandit*> vec_bandits;
    double ucb_param = 2;
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0., 0., 0.1, false, &ucb_param));
    }
    bandits.push_back(vec_bandits); 
    vec_bandits.clear();
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0.1, 0., 0.1));
    }
    bandits.push_back(vec_bandits); 
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    for (auto &v : bandits)
        for (auto p : v)
            delete p;
}

// for figure 2.5
void gradient_bandit(int num_bandits, int time_step)
{
    vector<vector<Bandit*>> bandits;
    vector<Bandit*> vec_bandits;
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0., 0., 0.1, false, NULL, true, true, 4));
    }
    bandits.push_back(vec_bandits); 
    vec_bandits.clear();
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0., 0., 0.1, false, NULL, true, false, 4));
    }
    bandits.push_back(vec_bandits); 
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0., 0., 0.4, false, NULL, true, true, 4));
    }
    bandits.push_back(vec_bandits); 
    vec_bandits.clear();
    for (int j=0;j<num_bandits;j++){
        vec_bandits.push_back(new Bandit(10, 0., 0., 0.4, false, NULL, true, false, 4));
    }
    bandits.push_back(vec_bandits); 
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    for (auto &v : bandits)
        for (auto p : v)
            delete p;

}

// figure 2.6
void figure2_6(int num_bandits, int time_step)
{
    vector<function<Bandit*(double*)>> generators ={
        [](double* epsilon) {return new Bandit(10, *epsilon, 0., 0.1, true, NULL, false, false, 0);},
        [](double* alpha) {return new Bandit(10, 0.0, 0., *alpha, true, NULL, true, true, 0);},
        [](double* coef) {return new Bandit(10, 0.0, 0., 0.1, false, coef, false, false, 0);},
        [](double* initial) {return new Bandit(10, 0.0, *initial, 0.1, false, NULL, false, false, 0);}
    };
    vector<vector<double>> parameters(4);
    for (int i=-7; i<-1;i++) parameters[0].push_back(i);
    for (int i=-5; i< 2;i++) parameters[1].push_back(i);
    for (int i=-4; i< 3;i++) parameters[2].push_back(i);
    for (int i=-2; i< 3;i++) parameters[3].push_back(i);

    vector<vector<Bandit*>> bandits;
    // construct bandits
    for (int i=0;i<4;i++){
        for (auto &x : parameters[i]){
            vector<Bandit*> vec_bandits;
            for (int j=0;j<num_bandits;j++){
                vec_bandits.push_back(generators[i](&x));
            }
            bandits.push_back(vec_bandits);
        }
    }
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    for (auto &v : bandits)
        for (auto p : v)
            delete p;
}

int main()
{
    figure2_1();
    epsilon_greedy(2000, 1000);
    optimistic_initial_values(2000, 1000);
    ucb(2000, 1000);
    gradient_bandit(2000, 1000);

    figure2_6(2000, 1000);
    return 0;
}
