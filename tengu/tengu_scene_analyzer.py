#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, json
import StringIO

class TenguSceneAnalyzer(object):

    def __init__(self, output_file=None):
        super(TenguSceneAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._output_file = output_file
        self._counter_dict = {}

    def analyze_scene(self, scene):
        """ analyze scene, and outputs a count report at intervals
        """

        for name in scene.flow_names:
            for named_flow in scene.named_flows(name):
                group = named_flow.group
                # check removed tracklets
                tracklets = named_flow.tracklets_by_dist()
                for tracklet in tracklets:
                    if tracklet.removed:
                        # increment
                        self.logger.info('found removed tracklet, counting {}'.format(tracklet))
                        # TODO
                        counts = self.tracklet_to_counts(tracklet)
                        if not self._counter_dict.has_key(group):
                            # TODO: the number of counting elements should be determined dynamically
                            # e.g. class0, class1, class2, speed
                            self._counter_dict[group] = [0 for i in range(len(counts))]
                        self._counter_dict[group] = [a + b for a, b in zip(self._counter_dict[group], counts)]
                        named_flow.remove_tracklet(tracklet)

        return self._counter_dict

    def tracklet_to_counts(self, tracklet):
        """ returns a list of numbers to be counted of this tracklet
        """
        # count 
        return [1, tracklet.speed]

    def finish_analysis(self):
        """ write its output to a file
        """
        if self._output_file is not None:
            f = open(self._output_file, 'w')
            sf = StringIO.StringIO()
            sorted_groups = sorted(self._counter_dict.keys())
            self.logger.info('sorted keys: {}'.format(sorted_groups))
            for g, sorted_group in enumerate(sorted_groups):
                js_array = json.dumps(self._counter_dict[sorted_group])
                csv = js_array[1:-1]
                self.logger.info('converted {} to {}'.format(js_array, csv))
                if g > 0:
                    sf.write(',')
                sf.write(csv)
            csv_line = sf.getvalue()
            self.logger.info('csv line = {}'.format(csv_line))
            sf.close()
            f.write(csv_line)