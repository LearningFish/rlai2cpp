
#include <iomanip>
#include <vector>
#include <utility>
#include <iostream>
#include <limits>
#include <algorithm>
#include <unordered_map>
using namespace std;

const int WORLD_SIZE = 5;
auto A_POS = make_pair(0, 1);
auto A_PRIME_POS = make_pair(4, 1);
auto B_POS = make_pair(0, 3);
auto B_PRIME_POS = make_pair(2, 3);
const double discount = 0.9;

vector<vector<double>> world(WORLD_SIZE, vector<double>(WORLD_SIZE, 0));

// left, up, right, down
vector<char> actions = { 'L','U','R','D' };

vector<vector<unordered_map<char, double>>> actionProb(WORLD_SIZE, vector<unordered_map<char, double>>(WORLD_SIZE, { { 'L',0.25 },{ 'U',0.25 },{ 'R',0.25 },{ 'D',0.25 } }));

vector<vector<unordered_map<char, pair<int, int>>>> nextState(WORLD_SIZE, vector<unordered_map<char, pair<int, int>>>(WORLD_SIZE));
vector<vector<unordered_map<char, double>>> actionReward(WORLD_SIZE, vector<unordered_map<char, double>>(WORLD_SIZE));

void states_reward()
{
	for (int i = 0; i<WORLD_SIZE; i++) {
		for (int j = 0; j<WORLD_SIZE; j++) {
			unordered_map<char, pair<int, int>> next;
			unordered_map<char, double> reward;

			next['U'] = i == 0 ? make_pair(i, j) : make_pair(i - 1, j);
			reward['U'] = i == 0 ? -1.0 : 0.0;

			next['D'] = i == WORLD_SIZE - 1 ? make_pair(i, j) : make_pair(i + 1, j);
			reward['D'] = i == WORLD_SIZE - 1 ? -1.0 : 0.0;

			next['L'] = j == 0 ? make_pair(i, j) : make_pair(i, j - 1);
			reward['L'] = j == 0 ? -1.0 : 0.0;

			next['R'] = j == WORLD_SIZE - 1 ? make_pair(i, j) : make_pair(i, j + 1);
			reward['R'] = j == WORLD_SIZE - 1 ? -1.0 : 0.0;

			if (make_pair(i, j) == A_POS) {
				next['L'] = next['R'] = next['D'] = next['U'] = A_PRIME_POS;
				reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0;
			}
			if (make_pair(i, j) == B_POS) {
				next['L'] = next['R'] = next['D'] = next['U'] = B_PRIME_POS;
				reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0;
			}
			nextState[i][j] = next;
			actionReward[i][j] = reward;
		}
	}
}

void draw_image(vector<vector<double>>& world_to_draw)
{
	cout << "==================================================" << endl;
	for (int i = 0; i<WORLD_SIZE; i++) {
		cout << "|";
		for (int j = 0; j<WORLD_SIZE; j++) {
			cout << " " << fixed << setprecision(2) << world_to_draw[i][j] << " |";
		}
		cout << endl;
	}
	cout << "==================================================" << endl;
}

// for figure 3.5
void cal_random_state_values()
{
	double delta = 1e8;
	while (delta>1e-4) {
		delta = 0;
		vector<vector<double>> new_world(WORLD_SIZE, vector<double>(WORLD_SIZE, 0));
		for (int i = 0; i<WORLD_SIZE; i++) {
			for (int j = 0; j<WORLD_SIZE; j++) {
				for (auto &action : actions) {
					auto newPosition = nextState[i][j][action];
					// Bellman equation for state value function
					new_world[i][j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * world[newPosition.first][newPosition.second]);
				}
				delta += abs(new_world[i][j] - world[i][j]);
			}
		}
		world = move(new_world);
		cout << "Random Policy: delta=" << delta << endl;
	}
	cout << "Random Policy" << endl;
	draw_image(world);
}

// for figure 3.8
void cal_optimal_state_values()
{
	// reset world
	world = vector<vector<double>>(WORLD_SIZE, vector<double>(WORLD_SIZE, 0));
	double delta = 1e8;
	while (delta>1e-4) {
		delta = 0;
		vector<vector<double>> new_world(WORLD_SIZE, vector<double>(WORLD_SIZE, 0));
		for (int i = 0; i<WORLD_SIZE; i++) {
			for (int j = 0; j<WORLD_SIZE; j++) {
				double max_value = -numeric_limits<double>::max();
				for (auto &action : actions) {
					auto newPosition = nextState[i][j][action];
					// Bellman optimility equation for state value function
					double state_value = actionReward[i][j][action] + discount * world[newPosition.first][newPosition.second];
					max_value = max(max_value, state_value);
				}
				new_world[i][j] = max_value;
				delta += abs(new_world[i][j] - world[i][j]);
			}
		}
		world = move(new_world);
		cout << "Optimal Policy: delta=" << delta << endl;
	}
	cout << "Optimal Policy" << endl;
	draw_image(world);
}

int main()
{
	states_reward();
	cal_random_state_values();
	cal_optimal_state_values();
	return 0;
}
