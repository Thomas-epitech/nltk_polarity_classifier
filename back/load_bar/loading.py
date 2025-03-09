def loading(iteration, total_iterations, prefix='', suffix='', decimals=1, length=100, left_side='[', right_side=']', loaded='#', unloaded='-'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total_iterations)))
    load = int(length * iteration // total_iterations)
    bar = loaded * load + unloaded * (length - load - 1)
    print(f"{prefix} {left_side}{bar}{right_side} {percent}% {suffix}", end='\r')
    if iteration >= total_iterations - 1:
        print("\n")