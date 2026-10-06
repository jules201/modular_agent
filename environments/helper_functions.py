def multivariate_gaussian(grid_size = 20, bounds = 4, height = 1, m1 = None, m2 = None, Sigma = None):
    """Return the multivariate Gaussian distribution on array pos."""

    # Our 2-dimensional distribution will be over variables X and Y
    X = np.linspace(-bounds, bounds, grid_size)
    Y = np.linspace(-bounds, bounds, grid_size)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    if m1 is None: # if not specifying means
      m1 = np.random.uniform(-bounds,bounds)
      m2 = np.random.uniform(-bounds,bounds)
    mu = np.array([m1, m2])

    # if Sigma is None: Sigma = dist.make_spd_matrix(2) # if not specifying covariance matrix, pick randomly

    height = np.random.uniform(1,height)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return height*norm(np.exp(-fac / 2) / N)

def norm(z):
  return z/z.sum()

def get_n_patches(grid_size = 20, bounds = 4, modes = 3, radius = None, var = 2):
# The distribution on the variables X, Y packed into pos.
  z = np.zeros((grid_size, grid_size))
  sigma = [[var,0],[0,var]]

  if radius is None: #randomly disperse
    for i in range(0,modes):
      z += multivariate_gaussian(grid_size = grid_size, bounds = bounds, Sigma = sigma)
    return z

  points = get_spaced_means(modes, radius)
  for i in range(0,modes):
    z += multivariate_gaussian(grid_size = grid_size, bounds = bounds, m1 = points[i][0], m2 = points[i][1])
  return z

def get_n_resources(grid_size = 20, bounds = 4, resources = 3, radius = None, offset = 0, var = 1):
# The distribution on the variables X, Y packed into pos.
  z = np.zeros((resources, grid_size, grid_size))

  sigma = [[var,0],[0,var]]

  if radius is None: #randomly disperse
    for i in range(0,resources):
      z[i,:,:] = multivariate_gaussian(grid_size = grid_size, bounds = bounds)
    return z

  points = get_spaced_means(resources, radius, offset)
  for i in range(resources):
    z[i,:,:] = multivariate_gaussian(grid_size = grid_size, bounds = bounds, m1 = points[i][0], m2 = points[i][1], Sigma = sigma)
  return z

# def KL_div(y):
#   return np.abs(0.05-distance.jensenshannon(patches1.flatten(),y.flatten()))

def get_spaced_means(modes, radius, offset):
    radians_between_each_point = 2*np.pi/modes
    list_of_points = []
    for p in range(0, modes):
        node = p + offset
        list_of_points.append( (radius*np.cos(node*radians_between_each_point),radius*np.sin(node*radians_between_each_point)) )
    return list_of_points


# patches1 = get_n_patches(grid_size = 20, bounds = 4, modes = 3, radius = None)
# patches2 = get_n_patches(grid_size = 20, bounds = 4, modes = 3, radius = None)

res = 4
z = get_n_resources(grid_size = 10, bounds = 3, resources = res, radius = 2, offset = 0, var = 2)