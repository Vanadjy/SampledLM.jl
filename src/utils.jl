function update_sample!(nls, k)
    if (length(nls.data_mem) / nls.nls_meta.nequ) + nls.sample_rate ≤ 1.0 #case where we don't have any data recovery
        #display("no recovery : $((length(nls.data_mem) / nls.nls_meta.nequ) + nls.sample_rate)")
        # creates a new sample which can select indexes which are not yet in data_mem
        nls.sample = sort(shuffle!(setdiff(collect(1:nls.nls_meta.nequ), nls.data_mem))[1:length(nls.sample)])

        #adding to data_mem the indexes contained in the current sample
        nls.data_mem = vcat(nls.data_mem, nls.sample)

    else #case where we have data recovery
        #display("recovery : $((length(nls.data_mem) / nls.nls_meta.nequ) + nls.sample_rate)")
        sample_size = Int(ceil(nls.sample_rate * nls.nls_meta.nequ))
        sample_complete = shuffle!(nls.data_mem)[1:(sample_size + length(nls.data_mem) - nls.nls_meta.nequ)]
        #picks up all the unvisited data and add a random part from the current memory
        nls.sample = sort(vcat(setdiff!(collect(1:nls.nls_meta.nequ), nls.data_mem), sample_complete))

        # adding in memory the sampled data used to complete the sample
        nls.data_mem = sample_complete
        push!(nls.epoch_counter, k)
    end
end

function uniform_sample(length, sample_rate)
    sample = []
    counter = 0.0
    for i in 2:length
        counter += sample_rate
        if counter ≥ 1.0
            push!(sample, i)
            counter -= 1.0
        end
    end
    sample
end

function basic_change_sample_rate(epoch_count::Int)
    if (epoch_count >= 0) && (epoch_count <= 1)
        return .05
      elseif (epoch_count > 1) && (epoch_count <= 2)
        return .3
      elseif (epoch_count > 2) && (epoch_count <= 4)
        return .7
      elseif (epoch_count > 4)
        return 1.0
      end
end

function basic_warn_sample_update(epoch_count::Int)
    if epoch_count ∈ [6, 11, 16]
        return true
    end
end

function get_filename(name::AbstractString)
    if name[(end - 2):end] == "bz2"
      filename = name
    elseif name[(end - 2):end] == "txt"
      filename = name * ".bz2"
    elseif name[(end - 2):end] == "pre"
      filename = name * ".txt.bz2"
    elseif occursin(r"[0-9]{3}", name[(end - 2):end])
      filename = name * "-pre.txt.bz2"
    else
      error("Cannot recognize $(name)")
    end
  
    return filename
  end