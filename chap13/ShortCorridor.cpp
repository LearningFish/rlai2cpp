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
# This is a reproduction of the plot shown in Example 13.1
# in Chapter 13, "Policy Gradient Methods". Book draft May 27, 2018.
# 
*/

#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

// True value of the first state
// Args:
//  p (float): probability of the action 'right'.
// Returns:
//  True value of the first state.
//  The expression is obtained by manually solving the easy linear system 
//  of Bellman equations using known dynamics.
double f(double p)
{
    return (2 * p - 4) / (p * (1 - p));
}
vector<double> f(vector<double>& p)
{
    vector<double> res(p.size());
    for (unsigned int i=0;i<p.size();i++)
        res[i] = f(p[i]);
    return res;
}

// generate even space vector
// start: start number
// end: end number
// num: total number in return vector
vector<double> linspace(double start, double end, int num)
{
    double d = (end-start)/num;
    vector<double> res(num);
    res[0] = start;
    res[num-1] = end;
    for (int i=1; i<=num-2; i++) res[i] = res[i-1] + d;
    return res;
}

int main()
{
    double epsilon = 0.05;

    // # Plot a graph 
    auto p = linspace(0.01, 0.99, 100);
    auto y = f(p);
    //ax.plot(p, y, color='red')

    //# Find a maximum point, can also be done analytically by taking a derivative
    auto imax = distance(y.begin(), max_element(y.begin(), y.end()));
    auto pmax = p[imax];
    auto ymax = y[imax];
    // ax.plot(pmax, ymax, color='green', marker="*", label="optimal point: f({0:.2f}) = {1:.2f}".format(pmax, ymax))

    //# Plot points of two epsilon-greedy policies
    //ax.plot(epsilon, f(epsilon), color='magenta', marker="o", label="epsilon-greedy left")
    //ax.plot(1 - epsilon, f(1 - epsilon), color='blue', marker="o", label="epsilon-greedy right")

    // ax.set_ylabel("Value of the first state")
    // ax.set_xlabel("Probability of the action 'right'")
    // ax.set_title("Short corridor with switched actions")
    // ax.set_ylim(ymin=-105.0, ymax=5)
    // ax.legend()
    // fig.tight_layout()
    // plt.show()

    return 0;
}