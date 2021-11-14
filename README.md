# MarketClearingPrice
https://github.com/yanzhipingliu/ClearingPrice

This package is to estimate the clearing price for each hour based on the Security Constrained Economic Dispatch (SCED) data and the demand data from independent system operators (ISO).  ISOs gather price/quantity bids from every generator in its territory for every hour of the day. The generators report the price at which they would be willing to sell various levels of generation. ISOs aggregate this Security Constrained Economic Dispatch (SCED) data into a system-wide supply curve, describing how much energy is available to the whole market at various prices. In each hour, ISOs dispatch generation in order to meet market demand in a manner that minimizes the costs to customers. The core of this tool involves estimating the system-wide supply curve, which describes the relationship between market price and market generation.
