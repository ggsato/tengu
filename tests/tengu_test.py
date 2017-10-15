#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tengu import Tengu, TenguObserver

def test_weak_ref_observers():
    """
    Check if, at least one weak ref object, is correctly garbage collected
    """

    weak_refs = 10

    tengu = Tengu()
    tengu.src = 'my_src'
    # create a strong ref object
    strong = TenguObserver()
    print('strong object = {}'.format(id(strong)))
    tengu.add_observer(strong)
    for i in range(weak_refs):
        weak = TenguObserver()
        print('weak object{} = {}'.format(i, id(weak)))
        tengu.add_observer(weak)

    keys = tengu._observers.keys()
    print('observer keys: {}'.format(keys))
    for key in keys:
        ref = tengu._observers[key]
        print('ref({}) -> {}'.format(id(ref), ref))

    assert len(tengu._observers) < weak_refs + 1