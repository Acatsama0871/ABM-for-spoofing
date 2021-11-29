# LOB.py
# Limit order book class


import uuid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sortedcontainers import SortedList


# order class
class Order:
    def __init__(self, price, size):
        self.ID = str(uuid.uuid4())  # order id
        self.price = price  # order price
        self.size = size  # order size when placing
        self.remaining_size = size  # the standing size in LOB

    def __str__(self):
        return f"ID: {self.ID}\nPrice: {self.price}\nSize: {self.size}\nRemaining Size: {self.remaining_size}"

    def __repr__(self):
        return f"ID: {self.ID} | Price: {self.price} | Size: {self.size} | Remaining Size: {self.remaining_size}"


class One_side_queue:
    def __init__(self, direction):
        self.direction = direction.lower()
        if direction == "bid":
            self.queue = SortedList([], key=lambda order: -order.price)
        elif direction == "ask":
            self.queue = SortedList([], key=lambda order: order.price)

    def add(self, *args):
        # add single order
        if len(args) == 1 and isinstance(args[0], Order):
            self.queue.add(args[0])
        # add a list of orders
        elif len(args) == 1 and isinstance(args[0], list):
            self.queue.update(args[0])
        else:
            raise InputError("Add: add an order or list of orders.")

    def remove(self, *args):
        # remove by id
        if len(args) == 1 and isinstance(args[0], str):
            # search
            index = None
            for i, cur_order in enumerate(self.queue):
                if cur_order.ID is args[0]:
                    index = i
            # remove
            if index is not None:
                self.queue.pop(index)
            else:
                raise OrderNotFoundError
        elif len(args) == 1 and isinstance(args[0], list):
            # search
            consumes = [
                cur_order for cur_order in self.queue if cur_order.ID in args[0]
            ]
            # remove
            if len(consumes) != len(args[0]):
                raise PartialFillError
            for cur_consume in consumes:
                self.queue.remove(cur_consume)
        else:
            raise InputError("Remove: input an order id or a list of order id.")

    def remove_empty(self):
        # find remaining size == 0
        consumes = [
            cur_order.ID for cur_order in self.queue if cur_order.remaining_size == 0
        ]
        # remove consumes
        self.remove(consumes)

    def inquire(self, ID):
        return any(cur_order.ID == ID for cur_order in self.queue)
    
    def inquire_remaining_size(self, ID):
        for cur_order in self.queue:
            if cur_order.ID == ID:
                return cur_order.remaining_size

    def __getitem__(self, index):
        return self.queue[index]

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        output = ""
        # header
        if self.direction == "bid":
            output += "      BID SIDE      \n"
        elif self.direction == "ask":
            output += "      ASK SIDE      \n"
        output += " QUANTITY    PRICE  \n"
        # order
        for cur_order in self.queue:
            output += f"  [{cur_order.size:.2f}  -  {cur_order.price:.2f}] \n"

        return output

    def __repr__(self):
        return self.__str__()


class LOB:
    def __init__(self):
        self.bid_side = One_side_queue(direction="bid")
        self.ask_side = One_side_queue(direction="ask")
    
    def get_best_bid_price(self):
        if len(self.bid_side) == 0:
            raise RunOutOfOrderError('bid')

        return self.bid_side[0].price

    def get_best_ask_price(self):
        if len(self.ask_side) == 0:
            raise RunOutOfOrderError('ask')

        return self.ask_side[0].price
    
    def get_mid_price(self):
        return (self.get_best_bid_price() + self.get_best_ask_price()) / 2
    
    def get_num_orders(self, direction):
        if direction.lower() == "bid":
            return len(self.bid_side)
        elif direction.lower() == "ask":
            return len(self.ask_side)
        else:
            raise SideKeyError
    
    def get_volume(self, direction):
        if direction.lower() == "bid":
            return sum(cur_order.remaining_size for cur_order in self.bid_side)
        elif direction.lower() == "ask":
            return sum(cur_order.remaining_size for cur_order in self.ask_side)
        else:
            raise SideKeyError

    def add(self, *args):
        if args[0].lower() == "bid":
            self.bid_side.add(args[1])
        elif args[0].lower() == "ask":
            self.ask_side.add(args[1])
        else:
            raise SideKeyError

    def remove(self, *args):
        if args[0].lower() == "bid":
            self.bid_side.remove(args[1])
        elif args[0].lower() == "ask":
            self.ask_side.remove(args[1])
        else:
            raise SideKeyError

    def inquire(self, direction, ID):
        if direction.lower() == "bid":
            found = self.bid_side.inquire(ID)
        elif direction.lower() == "ask":
            found = self.ask_side.inquire(ID)
        else:
            raise SideKeyError

        return found

    def matching(self):
        if len(self.bid_side) == 0 and len(self.ask_side) == 0:
            return

        for j in range(len(self.bid_side)):
            # matching
            filled = False

            for i in range(len(self.ask_side)):
                # stop condition: if the current bid has been filled
                if filled:
                    break
                # stop condition: if the current bid price is less than the current ask
                if self.bid_side[j].price < self.ask_side[i].price:
                    break

                # matching
                if self.bid_side[j].remaining_size > self.ask_side[i].remaining_size:
                    self.bid_side[j].remaining_size -= self.ask_side[i].remaining_size
                    self.ask_side[i].remaining_size = 0
                elif self.bid_side[j].remaining_size < self.ask_side[i].remaining_size:
                    self.ask_side[i].remaining_size -= self.bid_side[j].remaining_size
                    self.bid_side[j].remaining_size = 0
                    filled = True
                else:
                    self.bid_side[j].remaining_size = 0
                    self.ask_side[i].remaining_size = 0
                    filled = True

            # remove ask side
            self.ask_side.remove_empty()

        # remove from bid side
        self.bid_side.remove_empty()
    
    def random_generate(self, num_order=10, k=0.1, sig=0.1, random_size=1):
        # get current price
        if len(self.bid_side) == 0 or len(self.ask_side) == 0:
            cur_price = 100
        else:
            cur_price = (self.bid_side[0].price + self.ask_side[0].price) / 2
        
        # prices & sizes
        rng = np.random.default_rng()
        random_lognormal = rng.lognormal(0, sig, num_order * 2)
        bid_prices = np.round(cur_price * (1 - k) * (2.0 - random_lognormal[:num_order]), 2)
        ask_prices = np.round(cur_price * (1 + k) * random_lognormal[num_order:], 2)
        
        # add orders
        for cur_price in bid_prices:
            self.bid_side.add(Order(price=cur_price, size=random_size))
        for cur_price in ask_prices:
            self.ask_side.add(Order(price=cur_price, size=random_size))
    
    def price_level_dict(self, level=0):
        # bid side
        bid_level_dist = {}
        for cur_order in self.bid_side:
            cur_price = round(cur_order.price, level)
            cur_size = cur_order.remaining_size
            if cur_price in bid_level_dist:
                bid_level_dist[cur_price] += cur_size
            else:
                bid_level_dist[cur_price] = cur_size
        
        # ask size
        ask_level_dist = {}
        for cur_order in self.ask_side:
            cur_price = round(cur_order.price, level)
            cur_size = cur_order.remaining_size
            if cur_price in ask_level_dist:
                ask_level_dist[cur_price] += cur_size
            else:
                ask_level_dist[cur_price] = cur_size
        
        return bid_level_dist, ask_level_dist

    def __str__(self):
        output = "           LIMIT ORDER BOOK            \n"
        output += "      BID SIDE            ASK SIDE     \n"
        output += " QUANTITY    PRICE   PRICE    QUANTITY \n"

        len_difference = abs(len(self.ask_side) - len(self.bid_side))
        len_min = min(len(self.ask_side), len(self.bid_side))
        longer_side = "bid" if len(self.bid_side) > len(self.ask_side) else "ask"

        for i in range(len_min):
            output += f"  [{self.bid_side[i].remaining_size:.2f}  -  {self.bid_side[i].price:.2f}] | [{self.ask_side[i].price:.2f}  -  {self.ask_side[i].remaining_size:.2f}] \n"

        for j in range(len_min, len_min + len_difference):
            if longer_side == "bid":
                output += (
                    f"  [{self.bid_side[j].remaining_size:.2f}  -  {self.bid_side[j].price:.2f}] |"
                    + " " * 19
                    + "\n"
                )
            else:
                output += (
                    " " * 19
                    + f"| [{self.ask_side[j].price:.2f}  -  {self.ask_side[j].remaining_size:.2f}] \n"
                )

        return output

    def __repr__(self):
        return self.__str__()


# Exceptions
class OrderBookError(Exception):
    pass

class SideKeyError(OrderBookError):
    def __str__(self):
        return """Input side is neither bid nor ask."""

class PartialFillError(OrderBookError):
    def __str__(self):
        return """Some orders are not in LOB."""

class OrderNotFoundError(OrderBookError):
    def __str__(self):
        return """Order is not in LOB."""

class RunOutOfOrderError(OrderBookError):
    def __init__(self, side):
        self.side = side
    
    def __str__(self):
        return f"Run out of order on {self.side}."
class InputError(OrderBookError):
    def __init__(self, message="Invalid input"):
        super(InputError, self).__init__()
        self.message = message
    
    def __str__(self):
        return f"{self.message}"     

# utilities functions
# process list of book snapshots for hist plot
# single
def pre_BookPlot_single(book, level):
    # get price level dictionary
    bid, ask = book.price_level_dict(level=level)
    # bid
    bid_plot = []
    for cur_price in bid:
        for _ in range(bid[cur_price]):
            bid_plot.append(cur_price)
    # ask
    ask_plot = []
    for cur_price in ask:
        for _ in range(ask[cur_price]):
            ask_plot.append(cur_price)
    
    return bid_plot, ask_plot
# multiple
def pre_BookPlot(snapshots: list, level=2):    
    return [pre_BookPlot_single(i, level) for i in tqdm(snapshots)]

# plot order book
def plot_book(bid_data, ask_data, width=10):
    plt.hist(bid_data, label='BID')
    plt.hist(np.array(ask_data) + width, label='ASK')
    plt.xticks([])
    plt.legend()
    plt.show()
