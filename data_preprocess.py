from preprocess import batch_convert

#left 0-198
batch_convert('left', 0, 200)
#right 1-199
batch_convert('right', 1, 200)
#up 200-398
batch_convert('up', 200, 400)
#down 201-399
batch_convert('down', 201, 400)
#flick 400-598
batch_convert('flick', 400, 600)
#grab 401-599
batch_convert('grab', 401, 600)
#point 600-798
batch_convert('point', 600, 800)
