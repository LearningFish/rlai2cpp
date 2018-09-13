
/*
#######################################################################
# Copyright (C)                                                       #
# 2018 Donghai He(gsutilml@gmail.com)                                 #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
*/

#include <cmath>
#include <vector>
#include <random>
#include <unordered_map>
using namespace std;

// max # of cars at each location
const int MAX_CARS = 20;
// max # of cars to move during night
const int MAX_MOVE_OF_CARS = 5;
// expectation of rental requests at first location
const int RENTAL_REQUEST_FIRST_LOC  = 3;
// expectation of rental requests at second location
const int RENTAL_REQUEST_SECOND_LOC = 4;
// expectation of # of cars returned at first location
const int RETURNS_FIRST_LOC = 3;
// expectation of # of cars returned at second location
const int RETURNS_SECOND_LOC = 2;

const double DISCOUNT = 0.9;

// credit earned by one car
const double RENTAL_CREDIT = 10;
// cost of moving one car
const double MOVE_CAR_COST = 2;
// current policy
vector<vector<double>> policy(MAX_CARS+1, vector<double>(MAX_CARS+1, 0));
// current state value
vector<vector<double>> state_values(MAX_CARS+1, vector<double>(MAX_CARS+1, 0));
// all possible states
vector<pair<int,int>> states;
// all possible actions
vector<int> actions(2*MAX_MOVE_OF_CARS+1,0);

// factorial
unordered_map<int,long long> factorial_lookup = {{0,1},{1,1},{2,2}};
long long factorial(int n)
{
    if (factorial_lookup.find(n)==factorial_lookup.end())
        factorial_lookup[n] = n * factorial(n-1);
    return factorial_lookup[n];
}

// an up bound for poisson distribution
// if n is greater than this value, then the probability of getting n is truncated to 0
const int POISSON_UP_BOUND = 11;
// poisson pmf (lam--mean)
unordered_map<int,double> poisson_pmf_lookup;
double poisson_pmf(int k, int lam)
{
    int key = k * 10 + lam;
    if (poisson_pmf_lookup.find(key)==poisson_pmf_lookup.end())
        poisson_pmf_lookup[key] = exp(-lam) * pow(lam,k) / factorial(k);
    return poisson_pmf_lookup[key]; 
}

void init_states_actions()
{
    for (int i=0;i<MAX_CARS+1;i++){
        for (int j=0;j<MAX_CARS+1;j++){
            states.push_back(make_pair(i,j));
        }
    }
    for (int i=-MAX_MOVE_OF_CARS;i<MAX_MOVE_OF_CARS+1;i++)
        actions[i+MAX_MOVE_OF_CARS] = i;
}

// @state: [# of cars at 1st location, # of cars at 2nd location]
// @action: positive when moving cars from 1st to 2nd location
//          negative when moving cars from 2nd to 1st location
// @state_vals: state value matrix
double expected_return(pair<int,int>& state, int action, vector<vector<double>>& state_vals, bool constant_returned_cars)
{
    double total_return = -MOVE_CAR_COST * abs(action);
    // loop all possible rental requests
    for (int request_1st=0;request_1st<POISSON_UP_BOUND;request_1st++){
        for (int request_2nd=0;request_2nd<POISSON_UP_BOUND;request_2nd++){
            // moving cars
            int num_cars_1st = min (state.first - action, MAX_CARS);
            int num_cars_2nd = min (state.second + action, MAX_CARS);
            // rental requests should be less than actual # of cars
            int rental_1st = min (num_cars_1st, request_1st);
            int rental_2nd = min (num_cars_1st, request_2nd);
            // credits for renting
            double reward = (rental_1st+rental_2nd) * RENTAL_CREDIT;
            num_cars_1st -= rental_1st;
            num_cars_2nd -= rental_2nd;
            // probability for current combination of rental requests
            double prob = poisson_pmf(request_1st,RENTAL_REQUEST_FIRST_LOC)
                        * poisson_pmf(request_2nd,RENTAL_REQUEST_SECOND_LOC);

            if (constant_returned_cars){
                // get returned cars, those cars can be used for renting tomorrow
                int returned_cars_1st = RETURNS_FIRST_LOC;
                int returned_cars_2nd = RETURNS_SECOND_LOC;
                num_cars_1st = min(num_cars_1st + returned_cars_1st, MAX_CARS);
                num_cars_2nd = min(num_cars_2nd + returned_cars_2nd, MAX_CARS);
                total_return += prob * (reward + DISCOUNT * state_values[num_cars_1st][num_cars_2nd]);
            }else{
                for (int returned_cars_1st=0;returned_cars_1st<POISSON_UP_BOUND;returned_cars_1st++){
                    for (int returned_cars_2nd=0;returned_cars_2nd<POISSON_UP_BOUND;returned_cars_2nd++){
                        int num_cars_1st_ = min(num_cars_1st + returned_cars_1st, MAX_CARS);
                        int num_cars_2nd_ = min(num_cars_2nd + returned_cars_2nd, MAX_CARS);
                        double prob_ = poisson_pmf(returned_cars_1st,RETURNS_FIRST_LOC)
                                     * poisson_pmf(returned_cars_2nd,RETURNS_SECOND_LOC) * prob;
                        total_return += prob_ * (reward + DISCOUNT * state_values[num_cars_1st_][num_cars_2nd_]);
                    }
                }
            }
        }
    }
    return total_return;
}

void figure_4_2(bool constant_returned_cars=true)
{
    vector<vector<double>> value(MAX_CARS+1, vector<double>(MAX_CARS+1, 0.0));
    vector<vector<int>> policy(MAX_CARS+1, vector<int>(MAX_CARS+1, 0));

    int iteration  = 0;
    while (true){
        // policy iteration (in-place)
        while (true){
            auto new_value = value;
            double value_change = 0;
            for (int i=0;i<MAX_CARS+1;i++){
                for (int j=0;j<MAX_CARS+1;j++){
                    auto state = make_pair(i,j);
                    new_value[i][j] = expected_return(state, policy[i][j], new_value, constant_returned_cars);
                    value_change += abs(new_value[i][j] - value[i][j]);
                }
            }
            value = new_value;
            if (value_change<1e-4) break;
         }
         // policy improvement
        auto new_policy = policy;
        int policy_change = 0;
        for (int i=0;i<MAX_CARS+1;i++){
            for (int j=0;j<MAX_CARS+1;j++){
                double max_action_return = -numeric_limits<double>::max();
                int max_action_idx = 0;
                for (int r=0;r<actions.size();r++){
                    int action = actions[r];
                    double action_return;
                    if ((action>=0 && i>=action) || (action<0 && j>=abs(action))){
                        auto state = make_pair(i,j);
                        action_return = expected_return(state, action, value, constant_returned_cars);
                    }else{
                        auto state = make_pair(i,j);
                        action_return = -numeric_limits<double>::max();
                    }
                    if (max_action_return<action_return){
                        max_action_return = action_return;
                        max_action_idx = r;
                    }
                }
                new_policy[i][j] = actions[max_action_idx];
                policy_change += (int)(new_policy[i][j]!=policy[i][j]);
            }
        }
        policy = new_policy;
        if (policy_change==0) break;
        iteration++;
    } // end of while
}

int main()
{
    figure_4_2();
    return 0;
}
