from preprocess import batch_resize

batch_resize('left', 0, 200)
batch_resize('right', 1, 200)
batch_resize('up', 200, 400)
batch_resize('down', 201, 400)
batch_resize('flick', 400, 600)
batch_resize('grab', 401, 600)
batch_resize('point', 600, 800)