class State:
    def __init__(self, loc_a, loc_b, _id):
        self.loc_a = loc_a
        self.loc_b = loc_b
        self.id = _id

        self.action_prob = [[] for i in range(11)]

        for j in range(-5, 6):
            self.action_prob[j + 5] = [j, 1 / 11]

    def action(self, action):
        self.loc_a += action
        self.loc_b -= action

    def best_action(self):
        best_action_prob = [0, 0]
        for action_prob in self.action_prob:
            if action_prob[1] > best_action_prob[1]:
                best_action_prob = action_prob

        return best_action_prob[0]

    def print(self):
        print(f'{self.id}: {self.loc_a}, {self.loc_b}')

    def copy(self):
        return State(self.loc_a, self.loc_b, self.id)
