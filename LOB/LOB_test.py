# %%

from LOB import LOB, Order

# %%

# test matching
# the_book = LOB()
# bid_orders = [Order(price=90, size=3),
#               Order(price=110, size=3),
#               Order(price=100, size=2),
#               Order(price=80, size=2)]
# ask_orders = [Order(price=95, size=3),
#               Order(price=85, size=1),
#               Order(price=115, size=2),
#               Order(price=105, size=1)]
# the_book.add('bid', bid_orders)
# the_book.add('ask', ask_orders)
# print(the_book)
# BID
# SIDE
# QUANTITY
# PRICE
# [3.00 - 110.00]
# [2.00 - 100.00]
# [3.00 - 90.00]
# [2.00 - 80.00]
# ASK
# SIDE
# QUANTITY
# PRICE
# [1.00 - 85.00]
# [3.00 - 95.00]
# [1.00 - 105.00]
# [2.00 - 115.00]
# print('*' * 100)
# the_book.matching()
# print(the_book)
# BID
# SIDE
# QUANTITY
# PRICE
# [1.00 - 100.00]
# [3.00 - 90.00]
# [2.00 - 80.00]
# ASK
# SIDE
# QUANTITY
# PRICE
# [1.00 - 105.00]
# [2.00 - 115.00]

#%%
# # test inquires
# the_book = LOB()
# bid_orders = [Order(price=90, size=3),
#               Order(price=110, size=3),
#               Order(price=100, size=2),
#               Order(price=80, size=2)]
# ask_orders = [Order(price=95, size=3),
#               Order(price=85, size=1),
#               Order(price=115, size=2),
#               Order(price=105, size=1)]
# the_book.add('bid', bid_orders)
# the_book.add('ask', ask_orders)

# removed_order_id = bid_orders[0].ID
# the_book.remove('bid', removed_order_id)
# the_book.inquire('bid', removed_order_id)
# the_book.inquire('bid', bid_orders[1].ID)

# %%
# test random generate

#%%

import matplotlib.pyplot as plt
from LOB import LOB, Order

book = LOB()
book.random_generate(num_order=2000, k=0.05)
total_order = book.get_volume('bid') + book.get_volume('ask')
book.matching()
matching_order = book.get_volume('bid') + book.get_volume('ask') - total_order
bid, ask = book.price_level_dict()

bid_plot = []
for cur_price in bid:
    for _ in range(bid[cur_price]):
        bid_plot.append(cur_price)
ask_plot = []
for cur_price in ask:
    for _ in range(ask[cur_price]):
        ask_plot.append(cur_price)

plt.hist(bid_plot)
plt.show()
plt.hist(ask_plot)
plt.show()
print(matching_order)

#%%

book.random_generate(num_order=2000, k=0.05)
total_order = book.get_volume('bid') + book.get_volume('ask')
book.matching()
matching_order = book.get_volume('bid') + book.get_volume('ask') - total_order
bid, ask = book.price_level_dict()

bid_plot = []
for cur_price in bid:
    for _ in range(bid[cur_price]):
        bid_plot.append(cur_price)
ask_plot = []
for cur_price in ask:
    for _ in range(ask[cur_price]):
        ask_plot.append(cur_price)

plt.hist(bid_plot)
plt.show()
plt.hist(ask_plot)
plt.show()
print(matching_order)

