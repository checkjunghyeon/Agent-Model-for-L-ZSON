from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

import pandas
import numpy as np
class testPSL:
    def __init__(self):

        self.prob_array_room = [0.5,0.1,0.4,0.1,0.8,0.3,0.7,.0,.0]

        self.psl_model = PSLModel('objnav3')
        # Add Predicates
        self.add_predicates(self.psl_model)

        # Add Rules
        self.add_rules(self.psl_model)

        self.frontier_l6 = [[178, 300], [178, 301], [178, 302], [178, 303], [178, 304], [178, 307], [178, 308], [179, 289]]

        self.distances_16_inverse = [3.37298115e-01, 3.41149466e-01, 3.44977271e-01, 3.48779373e-01, 3.52548942e-01, 3.63601433e-01, 3.67189982e-01, 2.93572216e-01]
    def add_predicates(self, model):
        # if self.args.reasoning in ['both', 'obj']:
        #     predicate = Predicate('IsNearObj', closed=True, size=2)
        #     model.add_predicate(predicate)
        #
        #     predicate = Predicate('ObjCooccur', closed=True, size=1)
        #     model.add_predicate(predicate)
        # if self.args.reasoning in ['both', 'room']:
        predicate = Predicate('IsNearRoom', closed=True, size=2)
        model.add_predicate(predicate)

        predicate = Predicate('RoomCooccur', closed=True, size=1)
        model.add_predicate(predicate)

        predicate = Predicate('Choose', closed=False, size=1)
        model.add_predicate(predicate)

        predicate = Predicate('ShortDist', closed=True, size=1)
        model.add_predicate(predicate)

    def add_rules(self, model):
        # if self.args.reasoning in ['both', 'obj']:
        #     model.add_rule(Rule('1: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        #     model.add_rule(Rule('1: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        # if self.args.reasoning in ['both', 'room']:
        model.add_rule(Rule('1: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('1: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('1: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))

    def act(self):
        self.navigate_steps = 0

        if self.navigate_steps == 0:
            for predicate in self.psl_model.get_predicates().values():
                if predicate.name() in ['ROOMCOOCCUR']:
                    predicate.clear_data()

            prob_array_room_list = list(self.prob_array_room)
            data = pandas.DataFrame([[i, prob_array_room_list[i]] for i in range(len(prob_array_room_list))],
                                    columns=list(range(2)))
            self.psl_model.get_predicate('RoomCooccur').add_data(Partition.OBSERVATIONS, data)

        self.fbe()


    def fbe(self):

        ADDITIONAL_PSL_OPTIONS = {
            'log4j.threshold': 'INFO'
        }

        ADDITIONAL_CLI_OPTIONS = [
            # '--postgres'
        ]

        z = np.array(self.frontier_l6)
        scores = np.zeros((len(self.frontier_l6)))

        for predicate in self.psl_model.get_predicates().values():
            if predicate.name() in ['ISNEARROOM', 'CHOOSE', 'SHORTDIST']:
                predicate.clear_data()

        whether_near_room_list = [0.,0.6950585,0., 0., 0., 0.,0.,0.,0.]
        whether_near_room = np.array([0.,0.6950585,0., 0., 0., 0.,0.,0.,0.])

        for i, loc in enumerate(self.frontier_l6):
            data = pandas.DataFrame(
                [[j, i, whether_near_room_list[j]] for j in range(len(whether_near_room_list))],
                columns=list(range(3)))

            self.psl_model.get_predicate('IsNearRoom').add_data(Partition.OBSERVATIONS, data)

            score_1 = np.clip(1 - (1 - np.array(self.prob_array_room)) - (1 - whether_near_room), 0, 10)
            score_2 = 1 - np.clip(np.array(self.prob_array_room) + (1 - whether_near_room), -10, 1)
            scores[i] = np.sum(score_1) - np.sum(score_2)

        data = pandas.DataFrame([[i] for i in range( len(self.frontier_l6))], columns=list(range(1)))
        self.psl_model.get_predicate('Choose').add_data(Partition.TARGETS, data)

        data = pandas.DataFrame([[i, self.distances_16_inverse[i]] for i in range( len(self.frontier_l6))],
                                columns=list(range(2)))

        scores += 2 * np.array(self.distances_16_inverse)


        self.psl_model.get_predicate('ShortDist').add_data(Partition.OBSERVATIONS, data)

        result = self.psl_model.infer(additional_cli_options=ADDITIONAL_CLI_OPTIONS,
                                      psl_config=ADDITIONAL_PSL_OPTIONS)
        for key, value in result.items():
            result_dt_frame = value

        psl_scores = result_dt_frame.loc[:, 'truth']

        print(psl_scores)

        print(scores)

if __name__ == '__main__':
    psl = testPSL()
    psl.act()