"""
Represent a bundle adjustement problem in the form

    minimize   ½ ‖F(x)‖²

where `F(x)` is the vector of residuals.
"""
mutable struct SampledBAModel{T, S} <: AbstractNLSModel{T, S}
  # Meta and counters are required in every model
  meta::NLPModelMeta{T, S}
  # nls_meta
  nls_meta::NLSMeta{T, S}
  # Counters of NLPModel
  counters::NLSCounters
  # For each observation i, cams_indices[i] gives the index of thecamera used for this observation
  cams_indices::Vector{Int}
  # For each observation i, pnts_indices[i] gives the index of the 3D point observed in this observation
  pnts_indices::Vector{Int}
  # Each line contains the 2D coordinates of the observed point
  pt2d::S
  # Number of observations
  nobs::Int
  # Number of points
  npnts::Int
  # Number of cameras
  ncams::Int

  # temporary storage for residual
  k::S
  P1::S

  # temporary storage for jacobian
  JProdP321::Matrix{T}
  JProdP32::Matrix{T}
  JP1_mat::Matrix{T}
  JP2_mat::Matrix{T}
  JP3_mat::Matrix{T}
  P1_vec::S
  P1_cross::S
  P2_vec::S

  # sample features
  sample::AbstractVector{<:Integer}
  epoch_counter::AbstractVector{<:Integer}
  sample_rate::T
end

"""
    BundleAdjustmentModel(name::AbstractString; T::Type=Float64)

Constructor of BundleAdjustmentModel, creates an NLSModel with name `name` from a BundleAdjustment archive with precision `T`.
"""
function BAmodel_sto(name::AbstractString; T::Type = Float64, sample_rate = 1.0)
  filename = get_filename(name)
  filedir = fetch_ba_name(filename)
  path_and_filename = joinpath(filedir, filename)
  problem_name = filename[1:(end - 12)]

  cams_indices, pnts_indices, pt2d, x0, ncams, npnts, nobs = BundleAdjustmentModels.readfile(path_and_filename, T = T)

  S = typeof(x0)

  # variables: 9 parameters per camera + 3 coords per 3d point
  nvar = 9 * ncams + 3 * npnts
  # number of residuals: two residuals per 2d point
  nequ = 2 * nobs

  @debug "BundleAdjustmentModel $filename" nvar nequ

  meta = NLPModelMeta{T, S}(nvar, x0 = x0, name = problem_name)
  nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x0, nnzj = 2 * nobs * 12, nnzh = 0)

  k = similar(x0)
  P1 = similar(x0)

  JProdP321 = Matrix{T}(undef, 2, 12)
  JProdP32 = Matrix{T}(undef, 2, 6)
  JP1_mat = Matrix{T}(undef, 6, 12)
  JP2_mat = Matrix{T}(undef, 5, 6)
  JP3_mat = Matrix{T}(undef, 2, 5)
  P1_vec = S(undef, 3)
  P1_cross = S(undef, 3)
  P2_vec = S(undef, 2)

  sample_nobs = sort(randperm(nobs)[1:Int(ceil(sample_rate * nobs))])
  epoch_counter = [1]

  return SampledBAModel(
    meta,
    nls_meta,
    NLSCounters(),
    cams_indices,
    pnts_indices,
    pt2d,
    nobs,
    npnts,
    ncams,
    k,
    P1,
    JProdP321,
    JProdP32,
    JP1_mat,
    JP2_mat,
    JP3_mat,
    P1_vec,
    P1_cross,
    P2_vec,
    sample_nobs,
    epoch_counter,
    sample_rate,
  )
end

function NLPModels.residual!(nls::SampledBAModel, x::AbstractVector, rx::AbstractVector)
  nls.counters.:neval_residual += Int(floor(100 * nls.sample_rate))
  #increment!(nls, :neval_residual)
  residuals!(
    x,
    rx,
    nls.cams_indices,
    nls.pnts_indices,
    nls.nobs,
    nls.npnts,
    nls.k,
    nls.P1,
    nls.pt2d,
    nls.sample,
  )
  return rx
end

function residuals!(
  xs::AbstractVector,
  rxs::AbstractVector,
  cam_indices::Vector{Int},
  pnt_indices::Vector{Int},
  nobs::Int,
  npts::Int,
  ks::AbstractVector,
  Ps::AbstractVector,
  pt2d::AbstractVector,
  sample::AbstractVector,
)
  @simd for i in eachindex(sample)
    cam_index = cam_indices[sample[i]]
    pnt_index = pnt_indices[sample[i]]
    pnt_range = ((pnt_index - 1) * 3 + 1):((pnt_index - 1) * 3 + 3)
    cam_range = (3 * npts + (cam_index - 1) * 9 + 1):(3 * npts + (cam_index - 1) * 9 + 9)
    x = view(xs, pnt_range)
    c = view(xs, cam_range)
    k = view(ks, pnt_range)
    P = view(Ps, pnt_range)
    r = view(rxs, (2 * i - 1):(2 * i))
    projection!(x, c, r, k, P)
  end
  for j in sample
    rxs[(2 * j - 1):(2 * j)] .-= pt2d[(2 * j - 1):(2 * j)]
  end
end

function projection!(
  p3::AbstractVector,
  r::AbstractVector,
  t::AbstractVector,
  k1,
  k2,
  f,
  r2::AbstractVector,
  k::AbstractVector,
  P1::AbstractVector,
)
  θ = norm(r)
  k .= r ./ θ
  cross!(P1, k, p3)
  P1 .*= sin(θ)
  P1 .+= cos(θ) .* p3 .+ (1 - cos(θ)) .* dot(k, p3) .* k .+ t
  r2[1] = -P1[1] ./ P1[3]
  r2[2] = -P1[2] ./ P1[3]
  s = scaling_factor(r2, k1, k2)
  r2 .*= f * s
  return r2
end

projection!(x, c, r2, k, P1) =
  projection!(x, view(c, 1:3), view(c, 4:6), c[7], c[8], c[9], r2, k, P1)

function cross!(c::AbstractVector, a::AbstractVector, b::AbstractVector)
  if !(length(a) == length(b) == length(c) == 3)
    throw(DimensionMismatch("cross product is only defined for vectors of length 3"))
  end
  a1, a2, a3 = a
  b1, b2, b3 = b
  c[1] = a2 * b3 - a3 * b2
  c[2] = a3 * b1 - a1 * b3
  c[3] = a1 * b2 - a2 * b1
  c
end

function scaling_factor(point, k1, k2)
  sq_norm_point = dot(point, point)
  return 1 + sq_norm_point * (k1 + k2 * sq_norm_point)
end

function NLPModels.jac_structure_residual!(
  nls::SampledBAModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @simd for i in nls.sample
    idx_obs = (i - 1) * 24
    idx_cam = 3 * nls.npnts + 9 * (nls.cams_indices[i] - 1)
    idx_pnt = 3 * (nls.pnts_indices[i] - 1)

    # Only the two rows corresponding to the observation i are not empty
    p = 2 * i
    @views fill!(rows[(idx_obs + 1):(idx_obs + 12)], p - 1)
    @views fill!(rows[(idx_obs + 13):(idx_obs + 24)], p)

    # 3 columns for the 3D point observed
    @inbounds cols[(idx_obs + 1):(idx_obs + 3)] .= (idx_pnt + 1):(idx_pnt + 3)
    # 9 columns for the camera
    @inbounds cols[(idx_obs + 4):(idx_obs + 12)] .= (idx_cam + 1):(idx_cam + 9)
    # 3 columns for the 3D point observed
    @inbounds cols[(idx_obs + 13):(idx_obs + 15)] .= (idx_pnt + 1):(idx_pnt + 3)
    # 9 columns for the camera
    @inbounds cols[(idx_obs + 16):(idx_obs + 24)] .= (idx_cam + 1):(idx_cam + 9)
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  nls::SampledBAModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  nls.counters.:neval_jac_residual += Int(floor(100 * nls.sample_rate))
  #increment!(nls, :neval_jac_residual)
  T = eltype(x)

  fill!(nls.JP1_mat, zero(T))
  nls.JP1_mat[1, 7], nls.JP1_mat[2, 8], nls.JP1_mat[3, 9] = 1, 1, 1
  nls.JP1_mat[4, 10], nls.JP1_mat[5, 11], nls.JP1_mat[6, 12] = 1, 1, 1

  fill!(nls.JP2_mat, zero(T))
  nls.JP2_mat[3, 4], nls.JP2_mat[4, 5], nls.JP2_mat[5, 6] = 1, 1, 1

  @simd for i in nls.sample
    idx_cam = nls.cams_indices[i]
    idx_pnt = nls.pnts_indices[i]
    @views X = x[((idx_pnt - 1) * 3 + 1):((idx_pnt - 1) * 3 + 3)] # 3D point coordinates
    @views C = x[(3 * nls.npnts + (idx_cam - 1) * 9 + 1):(3 * nls.npnts + (idx_cam - 1) * 9 + 9)] # camera parameters
    @views r = C[1:3] # is the Rodrigues vector for the rotation
    @views t = C[4:6] # is the translation vector
    # k1, k2, f = C[7:9] is the focal length and radial distortion factors

    # JProdP321 = JP3∘P2∘P1 x JP2∘P1 x JP1
    P1!(r, t, X, nls.P1_vec, nls.P1_cross)
    P2!(nls.P1_vec, nls.P2_vec)
    JP2!(nls.JP2_mat, nls.P1_vec)
    JP1!(nls.JP1_mat, r, X, nls.P1_vec)
    JP3!(nls.JP3_mat, nls.P2_vec, C[9], C[7], C[8])
    mul!(nls.JProdP32, nls.JP3_mat, nls.JP2_mat)
    mul!(nls.JProdP321, nls.JProdP32, nls.JP1_mat)

    # Fill vals with the values of JProdP321 = [[∂P.x/∂X ∂P.x/∂C], [∂P.y/∂X ∂P.y/∂C]]
    # If a value is NaN, we put it to 0 not to take it into account
    replace!(nls.JProdP321, NaN => zero(T))
    @views vals[((i - 1) * 24 + 1):((i - 1) * 24 + 24)] = nls.JProdP321'[:]
  end
  return vals
end