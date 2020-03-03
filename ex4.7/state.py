class State:
    def __init__(self, loc_a, loc_b):
        self.loc_a = loc_a
        self.loc_b = loc_b

        self.total = self.loc_a + self.loc_b

        self.action_prob = [[] for i in range(11)]

        for j in range(-5, 6):
            self.action_prob[j+5] = [j, 1 / 11]

    def action(self, action):
        self.loc_a += action
        self.loc_b -= action

        self.total = self.loc_a + self.loc_b

    def print(self):
        print(f'{self.loc_a}, {self.loc_b}')
