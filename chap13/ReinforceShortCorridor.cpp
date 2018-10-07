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

#include <map>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <iostream>
#include <condition_variable>
#include <functional>
#include <algorithm>
using namespace std;

// Short corridor environment, see Example 13.1
class Env {
private:
	int s;
public:
	Env() { reset(); }
	void reset() { s = 0; }
	// Args:
	//     go_right (bool): chosen action
	// Returns:
	//     tuple of (reward, episode terminated?)
	pair<int, bool> step(bool go_right) {
		if (s == 0 || s == 2) {
			s = go_right ? s + 1 : max(0, s - 1);
		}
		else {
			//# self.s == 1
			s = go_right ? s - 1 : s + 1;
		}
		return (s == 3) ?
			//# terminal state
			make_pair(0, true) : make_pair(-1, false);
	}
};

class cutil {
public:
	static vector<double> col(const vector<vector<double>>& A, int col)
	{
		vector<double> r(A.size(), 0);
		for (unsigned int i = 0; i<r.size(); i++) r[i] = A[i][col];
		return r;
	}
	static vector<double> add(const vector<double>& x, const vector<double>& y)
	{
		auto r = x;
		for (unsigned int i = 0; i<r.size(); i++) r[i] += y[i];
		return r;
	}
	static vector<vector<double>> add_col(const vector<vector<double>>& A, int col, const vector<double>& y)
	{
		auto r = A;
		for (unsigned int i = 0; i<r.size(); i++) r[i][col] += y[i];
		return r;
	}
	static vector<double> exp(const vector<double>& x)
	{
		auto r = x;
		for (auto &v : r) v = std::exp(v);
		return r;
	}
	static vector<double> softmax(const vector<double>& x)
	{
		vector<double> t(x.size());
		double maxx = *max_element(x.begin(), x.end());
		double sum = 0;
		for (unsigned int i = 0; i<x.size(); i++) {
			t[i] = std::exp(x[i] - maxx);
			sum += t[i];
		}
		for (auto &y : t) y /= sum;
		return t;
	}
	// a - 1*1
	// v - 1*n
	static vector<double> dot(double a, const vector<double>& v) {
		auto r = v;
		for (auto &col : r) col = a * col;
		return r;
	}
	// a - 1*1
	// A - n*m
	static vector<vector<double>> dot(double a, const vector<vector<double>>& A) {
		auto r = A;
		for (auto &row : r) {
			for (auto &col : row)
				col = a * col;
		}
		return r;
	}
	// x - 1*n
	// y - n*1
	static double dot(const vector<double>& x, const vector<double>& y) {
		double r = 0;
		for (unsigned int i = 0; i<x.size(); i++)
			r += x[i] * y[i];
		return r;
	}
	// v - 1*n
	// A - n*m 
	// res - 1*m
	static vector<double> dot(const vector<double>& v, const vector<vector<double>>& A) {
		unsigned int n = A.size(), m = A.front().size();
		vector<double> res(m, 0);
		for (int col = 0; col<m; col++) {
			for (int row = 0; row<n; row++)
				res[col] += v[row] * A[row][col];
		}
		return res;
	}
	// A - n*m 
	// v - m*1
	// res - n*1
	static vector<double> dot(const vector<vector<double>>& A, const vector<double>& v) {
		unsigned int n = A.size(), m = A.front().size();
		vector<double> res(n, 0);
		for (int row = 0; row<n; row++)
			res[row] = dot(A[row], v);
		return res;
	}
	// A - n*m 
	// res - 1*m
	static vector<double> mean_col(const vector<vector<double>>& A) {
		unsigned int n = A.size(), m = A.front().size();
		vector<double> res(m, 0);
		for (int col = 0; col<m; col++) {
			for (int row = 0; row<n; row++)
				res[col] += A[row][col];
			res[col] /= n;
		}
		return res;
	}
};

// Agent that follows algorithm
// 'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
class Agent {
public:
	vector<double> theta;
	double alpha;
	double gamma;
	vector<vector<double>> x;
	vector<double> rewards;
	vector<double> actions;
	default_random_engine gen;
public:
	Agent(double _alpha, double _gamma) {
		//# set values such that initial conditions correspond to left-epsilon greedy
		theta = vector<double>({ -1.47, 1.47 });
		alpha = _alpha;
		gamma = _gamma;
		//# first column - left, second - right
		x = vector<vector<double>>({ { 0, 1 },{ 1, 0 } });
	}

	vector<double> get_pi() {
		auto h = cutil::dot(theta, x);
		double maxh = *max_element(h.begin(), h.end());
		vector<double> maxhv(h.size(), -maxh);
		auto addv = cutil::add(h, maxhv);
		auto t = cutil::exp(addv);
		double sumt = accumulate(t.begin(), t.end(), 0);
		auto pmf = cutil::dot(1 / sumt, t);
		//# never become deterministic, 
		//# guarantees episode finish
		int imin = distance(pmf.begin(), min_element(pmf.begin(), pmf.end()));
		double epsilon = 0.05;

		if (pmf[imin] < epsilon) {
			pmf.assign(pmf.size(), 1 - epsilon);
			pmf[imin] = epsilon;
		}
		return pmf;
	}

	double get_p_right() {
		return get_pi()[1];
	}

	bool choose_action(int reward, bool init_reward=false) {
		if (!init_reward)
			rewards.push_back(reward);
		auto pmf = get_pi();
		bernoulli_distribution bernouli(pmf[1]);
		bool go_right = bernouli(gen);
		actions.push_back(go_right);
		return go_right;
	}

	void episode_end(int last_reward) {
		rewards.push_back(last_reward);
		//# learn theta
		vector<double> G(rewards.size(), 0.0);
		*G.rbegin() = *rewards.rbegin();

		for (int i = G.size() - 1; i >= 1; i--)
			G[i] = gamma * G[i - 1] + rewards[i];

		double gamma_pow = 1;

		for (int i = 0; i<G.size(); i++) {
			auto j = actions[i] ? 1 : 0;

			auto pmf = get_pi();
			auto grad_lnpi = cutil::add(cutil::col(x, j), cutil::dot(-1.0, cutil::dot(x, pmf)));
			auto update = cutil::dot(alpha * gamma_pow * G[i], grad_lnpi);
			theta = cutil::add(theta, update);
			gamma_pow *= gamma;
		}

		rewards.clear();
		actions.clear();
	}
};

pair<vector<double>, vector<double>> trial(int num_episodes, double alpha, double gamma)
{
	Env env;
	Agent agent(alpha, gamma);

	vector<double> g1(num_episodes, 0.0);
	vector<double> p_right(num_episodes, 0.0);

	for (int episode_idx = 0; episode_idx<num_episodes; episode_idx++) {
		//# print("Episode {}".format(episode_idx))
		int rewards_sum = 0;
		int reward = -1;
		bool init_reward = true;
		env.reset();

		while (true) {
			auto go_right = agent.choose_action(reward, init_reward);
			init_reward = false;
			auto step_res = env.step(go_right);
			reward = step_res.first;
			auto episode_end = step_res.second;
			rewards_sum += reward;
			if (episode_end) {
				agent.episode_end(reward);
				//#print('rewards_sum: {}'.format(rewards_sum))
				//# decay alpha with time
				//#agent.alpha *= 0.995
				break;
			}
		}
		g1[episode_idx] = rewards_sum;
		p_right[episode_idx] = agent.get_p_right();
	}
	return make_pair(g1, p_right);
}

// a handy thread pool
class thread_pool
{
private:
	vector<thread> workers;
	queue<function<void()>> jobs;
	mutex job_mutex;
	condition_variable condition;
	bool stop = false;
public:
	thread_pool(int num_workers) {
		for (int i = 0; i<num_workers; i++)
			workers.push_back(thread(
				[&]() {
				while (!stop) {
					function<void()> job;
					{
						unique_lock<mutex> locker(job_mutex);
						condition.wait(locker, [&]()->bool {return stop || !jobs.empty(); });
						if (stop || jobs.empty()) continue;
						job = jobs.front();
						jobs.pop();
					}
					job();
				}
			}
		));
	}
	~thread_pool() {
		stop = true;
		condition.notify_all();
		for (auto &t : workers) 
			if (t.joinable()) t.join();
	}
	void add_job(function<void()> job)
	{
		{
			unique_lock<mutex> locker(job_mutex);
			jobs.push(job);
		}
		condition.notify_one();
	}
	int num_of_waiting_jobs() { return jobs.size(); }
};

void run() {
	int num_trials = 1000;
	int num_episodes = 1000;
	double alpha = 2e-4;
	double gamma = 1;

	vector<vector<double>> g1(num_trials, vector<double>(num_episodes, 0.0));
	vector<vector<double>> p_right(num_trials, vector<double>(num_episodes, 0.0));

	int num_workers = thread::hardware_concurrency();
	thread_pool pool(num_workers);
	map<int, pair<vector<double>, vector<double>>> trial_results_map;
	mutex mutex_trial_results_map;
	condition_variable condition;
	int trial_counter(0);
	for (int i = 0; i<num_trials; i++) {
		pool.add_job([&, i]() {
			int local_trial_idx = i;
			cout << "id=" << local_trial_idx << endl;
			auto res = trial(num_episodes, alpha, gamma);
			{
				unique_lock<mutex> locker(mutex_trial_results_map);
				trial_results_map[local_trial_idx] = res;
				trial_counter++;
				if (trial_counter == num_trials) condition.notify_one();
			}
		});
	}

	{
		unique_lock<mutex> locker(mutex_trial_results_map);
		condition.wait(locker);
	}

	for (auto &kvp : trial_results_map) {
		int trial_idx = kvp.first;
		cout << "Trial {" << trial_idx << "}" << endl;
		auto &out = kvp.second;
		g1[trial_idx] = out.first;
		p_right[trial_idx] = out.second;
	}
	auto avg_rewards_sum = cutil::mean_col(g1);
	auto avg_p_right = cutil::mean_col(p_right);
	/*
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
	*/
}

int main()
{
	run();
	return 0;
}
