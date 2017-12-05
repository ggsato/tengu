#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, json
import StringIO

class TenguCountItem(object):

    def __init__(self, header, value):
        super(TenguCountItem, self).__init__()
        self.header = header
        self.value = value

class TenguSumItem(TenguCountItem):
    pass

class TenguAvgItem(TenguCountItem):
    pass

class TenguSceneAnalyzer(object):

    def __init__(self, output_file=None):
        super(TenguSceneAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._output_file = output_file
        self._counter_dict = {}

    def analyze_scene(self, scene):
        """ analyze scene, and outputs a count report at intervals
        GROUP A     | GROUP A     | ... | GROUP Z  
        CLASS 0     | CLASS N     | ... | CLASS N 
        cA0-1 cA0-2 | cAN-1 cAN-2 | ... | cZN-1 cZN-2
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
                        count_items = self.tracklet_to_count_items(tracklet)
                        class_name = tracklet.class_name
                        if not self._counter_dict.has_key(group):
                            # TODO: the number of counting elements should be determined dynamically
                            # e.g. class0, class1, class2, speed
                            self._counter_dict[group] = {}
                        if not self._counter_dict[group].has_key(class_name):
                            self._counter_dict[group][class_name] = []
                        self._counter_dict[group][class_name].append(count_items)
                        named_flow.remove_tracklet(tracklet)

        return self._counter_dict

    def tracklet_to_count_items(self, tracklet):
        """ returns a list of numbers to be counted of this tracklet
        """
        # count 
        return [TenguSumItem('Count', 1), TenguAvgItem('Speed', tracklet.speed)]

    def finish_analysis(self):
        """ write its output to a file
        """
        if self._output_file is not None:
            self.logger.info('writing a report to {}'.format(self._output_file))
            f = open(self._output_file, 'w')
            header = None
            row = StringIO.StringIO()
            sorted_groups = sorted(self._counter_dict.keys())
            self.logger.info('groups: {}'.format(sorted_groups))
            for g, sorted_group in enumerate(sorted_groups):
                for c, class_name in enumerate(self._counter_dict[sorted_group]):
                    count_items_list = self._counter_dict[sorted_group][class_name]
                    total_count_values = [0 for i in range(len(count_items_list[0]))]
                    for count_items in count_items_list:
                        for i, count_item in enumerate(count_items):
                            if isinstance(count_item, TenguAvgItem):
                                value = float(count_item.value) / len(count_items_list)
                            else:
                                value = count_item.value
                            total_count_values[i] += value
                    # got totals
                    for t, total_count_value in enumerate(total_count_values):
                        if header is None:
                            # this is the first item being written
                            header = StringIO.StringIO()
                        else:
                            header.write(',')
                            row.write(',')
                        # write
                        header.write('{}-{}-{}'.format(sorted_group, class_name, count_items_list[0][t].header))
                        if isinstance(total_count_value, float):
                            row.write('{:03.2f}'.format(total_count_value))
                        else:
                            row.write('{:d}'.format(total_count_value))

            csv_hedaer = header.getvalue()
            csv_row = row.getvalue()
            self.logger.info('csv header = {}'.format(csv_hedaer))
            self.logger.info('csv row = {}'.format(csv_row))
            header.close()
            row.close()
            f.write('{}\n{}'.format(csv_hedaer, csv_row))
            f.close()