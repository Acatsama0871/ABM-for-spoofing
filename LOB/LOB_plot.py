#%%
from LOB import LOB, pre_BookPlot, plot_book

if __name__ == '__main__':
    # parameters
    num_times = 5

    # initialize
    book = LOB()

    # observer
    price_series = []
    num_matching_series = []
    book_snapshots = []

    #%%
    print('Book start')
    for _ in range(num_times):
        # generate new order
        book.random_generate(num_order=2000, k=0.05)
        order_vol_temp = book.get_volume('bid') + book.get_volume('ask')
        # matching
        book.matching()
        num_matching = order_vol_temp - book.get_volume('bid') + book.get_volume('ask')
        # snapshot
        book_snapshots.append(book)
        # add to observer
        num_matching_series.append(num_matching)   
        price_series.append(book.get_mid_price()) 

    # %%
    # process
    print('Processing start')
    process_result = pre_BookPlot(book_snapshots)

    # %%
    # plot
    print('Plot start')
    for cur_bid, cur_ask in process_result:
        plot_book(cur_bid, cur_ask)