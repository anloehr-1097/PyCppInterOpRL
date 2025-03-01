#include <Eigen/Dense>
#include <Eigen/Core>

#include "test.h"


int main(){
    Policy pol = Policy();
    Policy critic = Policy();

    Eigen::VectorXd state(8), next_state(8);
    int action = 3;
    double reward = 0.47;

    // state = Eigen::VectorXd::Random(4);
    state << 1,2,3,4,5,6,7,8;
    next_state << 1,2,3,4,5,6,7,8;
    auto mdp_trans = MDPTransition(state, action, reward, next_state);

    ReplayBuffer rp = ReplayBuffer();
    rp.append_to_buffer(mdp_trans);

    learn(rp, pol, critic, 4);

    return 0;
}
