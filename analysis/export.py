#!/usr/bin/env python
# Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import manage

def tune(queue, table, operator, N):
    confs = list()
    if operator.casefold() == "max" or operator.casefold() == "min":
        m_range = manage.get_M_range(queue, table, N)
        for m in m_range:
            queue.execute("SELECT itemsPerBlock,GBS,time,time_err,cov FROM " + table + " WHERE (GBS = (SELECT " + operator + "(GBS) FROM " + table + " WHERE (M = " + str(m[0]) + " AND N = " + N + ")) AND (M = " + str(m[0]) + " AND N = " + N + "))")
            best = queue.fetchall()
            confs.append([m[0], best[0][0], best[0][1], best[0][2], best[0][3], best[0][4]])
    return confs

