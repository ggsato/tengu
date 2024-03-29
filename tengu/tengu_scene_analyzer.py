#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, json
from io import open
from io import StringIO 
from pandas import DataFrame
import pandas as pd

class TenguSceneAnalyzer(object):

    def __init__(self, output_file=None, ignore_default=True):
        super(TenguSceneAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._output_file = output_file
        self._ignore_default = ignore_default
        self._df = None

    def analyze_scene(self, scene):
        """ turn removed tracklets into counts
        GROUP A     | GROUP A     | ... | GROUP Z  
        CLASS 0     | CLASS N     | ... | CLASS N 
        cA0-1 cA0-2 | cAN-1 cAN-2 | ... | cZN-1 cZN-2
        """
        # {'group': [vehicle_dicts]}
        counted_tracklets = {}

        return counted_tracklets

    def count_tracklet(self, tracklet, group):
        self.logger.debug('found removed tracklet, counting {}'.format(tracklet))
        count_dict = self.tracklet_to_count_dict(tracklet)
        if count_dict is not None:
            count_dict['Class'] = {tracklet.obj_id: tracklet.class_name}
            count_dict['Group'] = {tracklet.obj_id: group}
            df = DataFrame.from_dict(count_dict)
            self.logger.debug('df = {}'.format(df))
            # merge
            if self._df is None:
                self._df = df
            else:
                self._df = self._df.merge(df, how='outer')
                self.logger.debug('self df = {}'.format(self._df))

    def tracklet_to_count_dict(self, tracklet):
        """ returns a list of numbers to be counted of this tracklet
        override as necessary
        """
        # count 
        count_dict = {}
        count_dict['Count'] = {tracklet.obj_id: 1}
        # note that this speed is UNIT/FRAME
        count_dict['Speed'] = {tracklet.obj_id: tracklet.speed}
        return count_dict

    def finish_analysis(self):
        """ write its output to a file
        override as necessary
        """
        if self._output_file is not None and self._df is not None:
            self.logger.info('writing a report to {}'.format(self._output_file))
            
            grouped_count = self._df['Count'].groupby([self._df['Group'], self._df['Class']]).sum()
            self.logger.info(grouped_count)

            grouped_speed = self._df['Speed'].groupby([self._df['Group']]).mean()
            self.logger.info(grouped_speed)

            header = None
            row = None
            for group in grouped_speed.index:
                self.logger.info('checking {}'.format(group))
                if group == 'default' and self._ignore_default:
                    self.logger.info('skipping default')
                    continue
                grouped_by_class = grouped_count[group]
                for class_name in grouped_by_class.index:
                    self.logger.info('{}, {}, {}'.format(group, class_name, grouped_count[group, class_name]))
                    if header is None:
                        # this is the first time
                        header = StringIO()
                        row = StringIO()
                    else:
                        header.write(u',')
                        row.write(u',')

                    header.write(u'{}-{}'.format(group, class_name))
                    row.write(u'{}'.format(grouped_count[group, class_name]))
                # add group level average speed
                header.write(u',{}-Speed'.format(group))
                row.write(u',{}'.format('{:03.2f}'.format(grouped_speed[group])))

            if header is not None:

                header_value = header.getvalue()
                header.close()
                row_value = row.getvalue()
                row.close()

                f = open(self._output_file, 'w', encoding='utf-8')
                f.write(u'{}\n'.format(header_value))
                f.write(u'{}\n'.format(row_value))
                f.close()

        self.reset_counter()

    def reset_counter(self):
        self.logger.info('resetting counter...')
        self._df = None