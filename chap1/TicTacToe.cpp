// TicTacToe.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <random>
#include <iostream>
#include <functional>
#include <unordered_map>
using namespace std;

const int BOARD_DIM = 3;
const int BOARD_SIZE = BOARD_DIM*BOARD_DIM;
const int BlankCell = 0;
const int PlayerX = 1;
const int PlayerO = -1;
const int WinnerX = 1;
const int WinnerO = -1;
const int Tie = 0;

static unsigned int initial_state_hash = -1;
default_random_engine global_generator;

class State {
private:
	bool done_;
	int win_;
	unsigned int hash_;
	vector<int> board_;
public:
	State() = default;
	State(const vector<int>& board) :done_(false), win_(false) { board_ = board; hash_ = hash(board_); }
	State(const vector<int>&& board) :done_(false), win_(false) { board_ = move(board); hash_ = hash(board_); }
	State(const State&& t) { done_ = t.done_; win_ = t.win_; board_ = move(t.board_); hash_ = t.hash(); }
	State& operator = (const State&& t) { if (this != &t) { done_ = t.done_; win_ = t.win_; board_ = move(t.board_); hash_ = t.hash(); } return *this; }
	static int hash(const vector<int>& vec) {
		unsigned int seed = 0;
		for (auto &x : vec) seed ^= std::hash<int>{}(x)+0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}
	inline bool done() const { return done_; }
	inline void done(bool d) { done_ = d; }
	inline void win(int w) { win_ = w; }
	inline int  win() const { return win_; }
	inline unsigned int hash() const { return hash_; }
	inline int value(int row, int col) const { return board_[rowcol2pos(row, col)]; }
	inline int value(int pos) const { return board_[pos]; }
	static inline int rowcol2pos(int row, int col) { return row*BOARD_DIM + col; }
	static inline pair<int, int> pos2rowcol(int pos) { return make_pair(pos / BOARD_DIM, pos%BOARD_DIM); }
	static State create_next(const State& t, int row, int col, int player) {
		auto new_board = t.board_;
		new_board[rowcol2pos(row, col)] = player;
		return State(move(new_board));
	}
	static int next_state_hash(const State& t, int row, int col, int player) {
		auto player_backup = t.value(row, col);
		const_cast<State&>(t).board_[rowcol2pos(row, col)] = player;
		auto hash_value = hash(t.board_);
		const_cast<State&>(t).board_[rowcol2pos(row, col)] = player_backup;
		return hash_value;
	}
	static pair<bool,int> check_done_win(const State& t) {
		vector<int> scores;
		//rows, cols
		for (int r = 0; r<BOARD_DIM; r++) {
			int rscore = 0, cscore = 0;
			for (int c = 0; c<BOARD_DIM; c++) {
				rscore += t.value(r, c);
				cscore += t.value(c, r);
			}
			scores.push_back(rscore);
			scores.push_back(cscore);
		}
		//diag
		int dscore = 0, rdscore = 0;
		for (int r = 0; r<BOARD_DIM; r++) {
			dscore += t.value(r, r);
			rdscore += t.value(r, BOARD_DIM - 1 - r);
		}
		scores.push_back(dscore);
		scores.push_back(rdscore);
		for (auto score : scores) {
			if (score == BOARD_DIM) return make_pair(true, WinnerX);
			if (score == -BOARD_DIM) return make_pair(true, WinnerO);
		}
		int tot = 0;
		for (auto x : t.board_) {
			tot += abs(x);
		}
		if (tot == BOARD_SIZE) return make_pair(true, Tie); //Tie
		return make_pair(false, Tie); // gaming is not done.
	}
	static unordered_map<unsigned int, State> create_all_states() {
		unordered_map<unsigned int, State> all_states;
		// lambda function
		function<void(const State&, int)> create_from_current = [&](const State& current_state, int player) {
			for (int r = 0; r<BOARD_DIM; r++) {
				for (int c = 0; c<BOARD_DIM; c++) {
					if (current_state.value(r, c) == BlankCell) {
						auto next_state = State::create_next(current_state, r, c, player);
						auto done_win = State::check_done_win(next_state);
						next_state.done(done_win.first);
						next_state.win(done_win.second);
						auto next_hash_value = next_state.hash();
						all_states[next_hash_value] = move(next_state);
						const auto &next_state_ref = all_states[next_hash_value];
						if (!next_state_ref.done()) create_from_current(next_state_ref, -player);
					}
				}
			}
		};
		State init_state(vector<int>(BOARD_SIZE, BlankCell));
		initial_state_hash = init_state.hash();
		all_states[initial_state_hash] = move(init_state);
		create_from_current(all_states[initial_state_hash], PlayerX);
		return all_states;
	}
};
ostream& operator << (ostream& out, const State& t)
{
	for (int r = 0; r<BOARD_DIM; r++) {
		out << "-------------" << endl;
		string line = "|";
		for (int c = 0; c<BOARD_DIM; c++) {
			switch (t.value(r, c)) {
			case BlankCell: line += " "; break;
			case PlayerX: line += "X"; break;
			case PlayerO: line += "O"; break;
			};
			line += "|";
		}
		out << line.c_str() << endl;
	}
	out << "-------------" << endl;
	return out;
}

// Player
class Player {
public:
	virtual void reset() = 0;
	virtual int  role() = 0;
	virtual void role(int role) = 0;
	virtual void state(const State* pstate) = 0;
	virtual void reward(double r) = 0;
	virtual vector<int> action() = 0;
	virtual unordered_map<unsigned int, double> value_table() const = 0;
	virtual void value_table(const unordered_map<unsigned int, double>& vtable) = 0;
};

class AIPlayer : public Player {
private:
	int role_; // X or O
	unordered_map<unsigned int, State>* all_states_ptr_;
	unordered_map<unsigned int, double> value_table_; //[state hash, state value]
	double step_size_;
	double explore_rate_;
	vector<const State*> state_ptrs_;
	uniform_real_distribution<double> unif_real_;
public:
	AIPlayer(int arole, unordered_map<unsigned int, State>* all_states, double step_size = 0.1, double explore_rate = 0.1)
		:step_size_(step_size), explore_rate_(explore_rate), all_states_ptr_(all_states) {
		role(arole);
		unif_real_.param(uniform_real_distribution<double>::param_type(0.0,1.0));
	}
	void reset() { state_ptrs_.clear(); }
	int role() { return role_; }
	void role(int role)
	{
		role_ = role;
		// initialize value table
		for (auto &kv : *all_states_ptr_) {
			if (kv.second.done())
				value_table_[kv.first] = kv.second.win() == role_ ? 1 : 0;
			else
				value_table_[kv.first] = 0.5;
		}
	}
	void state(const State* pstate) { state_ptrs_.push_back(pstate); }
	void reward(double r)
	{
		// update reward, learning
		if (state_ptrs_.empty()) return;
		double target = r;
		for (auto rit = state_ptrs_.rbegin(); rit != state_ptrs_.rend(); ++rit) {
			auto lastest_state = *rit;
			value_table_[lastest_state->hash()] += step_size_ * (target - value_table_[lastest_state->hash()]);
			target = value_table_[lastest_state->hash()];
		}
		state_ptrs_.clear();
	}
	vector<int> action()
	{
		auto latest_state = state_ptrs_.back();
		vector<const State*> possible_state_ptrs;
		vector<pair<int, int>> possible_positions;
		for (int r = 0; r<BOARD_DIM; r++) {
			for (int c = 0; c<BOARD_DIM; c++) {
				if (latest_state->value(r, c) == BlankCell) {
					auto hash_value = State::next_state_hash(*latest_state, r, c, role_);
					possible_state_ptrs.push_back(&((*all_states_ptr_)[hash_value]));
					possible_positions.push_back(make_pair(r, c));
				}
			}
		}
		// check random pick
		if (unif_real_(global_generator)<explore_rate_) {
			uniform_int_distribution<int> unif_int(0, possible_positions.size() - 1);
			auto pos = unif_int(global_generator);
			state_ptrs_.clear();
			return vector<int>({ possible_positions[pos].first,possible_positions[pos].second,role_ });
		}

		// pick the largest value
		int max_pos = -1;
		double max_value = INT64_MIN;
		for (unsigned int i = 0; i<possible_state_ptrs.size(); i++) {
			auto ps = possible_state_ptrs[i];
			double value = value_table_[ps->hash()];
			if (max_value<value) {
				max_value = value;
				max_pos = i;
			}
		}
		return vector<int>({ possible_positions[max_pos].first,possible_positions[max_pos].second,role_ });
	}
	unordered_map<unsigned int, double> value_table() const { return value_table_; }
	void value_table(const unordered_map<unsigned int, double>& vtable) { value_table_ = vtable; }
};

class HumanPlayer : public Player {
private:
	int role_;
	const State* current_state_;
public:
	HumanPlayer(int arole) :current_state_(NULL) { role(arole); }
	void reset() { current_state_ = NULL; }
	int  role() { return role_; }
	void role(int role) { role_ = role; }
	void state(const State* pstate) { current_state_ = pstate; }
	void reward(double r) {}
	vector<int> action()
	{
		int pos;
		do {
			// Ask human to input
			cout << "Input your position:"; cin >> pos;
		} while (current_state_->value(pos) != BlankCell);
		auto rc = State::pos2rowcol(pos);
		return vector<int>({ rc.first,rc.second,role_ });
	}
	unordered_map<unsigned int, double> value_table() const { return unordered_map<unsigned int, double>(); }
	void value_table(const unordered_map<unsigned int, double>& vtable) { }
};

// Judge
class Judge {
private:
	Player* player1_;
	Player* player2_;
	Player* current_player_;
	bool feedback_;
	const State* current_state_;
	unordered_map<unsigned int, State>* all_states_ptr_;
public:
	Judge(unordered_map<unsigned int, State>* all_states, Player* player1, Player* player2, bool feedback = true)
		:all_states_ptr_(all_states), player1_(player1), player2_(player2), feedback_(feedback), current_player_(nullptr)
	{
		current_state_ = &((*all_states_ptr_)[initial_state_hash]);
	}
	void reward()
	{
		if (current_state_->win() == player1_->role()) {
			player1_->reward(1);
			player2_->reward(0);
		}
		else if (current_state_->win() == player2_->role()) {
			player1_->reward(0);
			player2_->reward(1);
		}
		else {
			player1_->reward(0.1);
			player2_->reward(0.5);
		}
	}
	void feed_current_state()
	{
		player1_->state(current_state_);
		player2_->state(current_state_);
	}
	void reset()
	{
		player1_->reset();
		player2_->reset();
		current_player_ = nullptr;
		current_state_ = &((*all_states_ptr_)[initial_state_hash]);
	}
	int play(bool show = false)
	{
		reset();
		feed_current_state();
		while (true) {
			current_player_ = current_player_ == player1_ ? player2_ : player1_;
			if (show) cout << *current_state_;
			auto action = current_player_->action();
			current_state_ = &((*all_states_ptr_)[State::next_state_hash(*current_state_, action[0], action[1], action[2])]);
			feed_current_state();
			if (current_state_->done()) {
				if (feedback_) reward();
				return current_state_->win();
			}
		}
	}
};

class TicTacToe
{
private:
	unordered_map<unsigned int, State> all_states_;
	unordered_map<int, unordered_map<unsigned int, double> > value_tables; //[player role, [state hash, estimated value]]
public:
	TicTacToe() { all_states_ = State::create_all_states(); }
	void train(int epochs = 20000)
	{
		AIPlayer player1(PlayerX, &all_states_), player2(PlayerO, &all_states_);
		Judge judge(&all_states_, &player1, &player2);
		double player1_win = 0, player2_win = 0;
		for (int i = 0; i<epochs; i++) {
			if (i % 100 == 0) cout << "Epoch " << i << ": " << endl;
			int winner = judge.play();
			if (winner == player1.role()) player1_win += 1;
			if (winner == player2.role()) player2_win += 1;
			judge.reset();
		}
		cout << "Player 1 Win : " << player1_win / epochs << endl;
		cout << "Player 2 Win : " << player2_win / epochs << endl;
		value_tables[player1.role()] = player1.value_table();
		value_tables[player2.role()] = player2.value_table();
	}
	void compete(int turns = 500)
	{
		AIPlayer player1(PlayerX, &all_states_, 0.1, 0), player2(PlayerO, &all_states_, 0.1, 0);
		Judge judge(&all_states_, &player1, &player2, false);
		player1.value_table(value_tables[player1.role()]);
		player2.value_table(value_tables[player2.role()]);
		double player1_win = 0, player2_win = 0;
		for (int i = 0; i<turns; i++) {
			if (i % 100 == 0) cout << "Epoch " << i << ": " << endl;
			int winner = judge.play();
			if (winner == player1.role()) player1_win += 1;
			if (winner == player2.role()) player2_win += 1;
			judge.reset();
		}
		cout << "Player 1 Win : " << player1_win / turns << endl;
		cout << "Player 2 Win : " << player2_win / turns << endl;
	}
	void play()
	{
		while (true) {
			AIPlayer ai_player(PlayerX, &all_states_, 0.1, 0);
			HumanPlayer man_player(PlayerO);
			Judge judge(&all_states_, &ai_player, &man_player, false);
			ai_player.value_table(value_tables[ai_player.role()]);
			int winner = judge.play(true);
			if (winner == ai_player.role()) cout << " Lose !" << endl;
			else if (winner == man_player.role()) cout << " Win !" << endl;
			else cout << " Tie !" << endl;
		}
	}
};

int main()
{
	TicTacToe tic;
	tic.train(100000);
	tic.compete(1000);
	tic.play();
	return 0;
}
