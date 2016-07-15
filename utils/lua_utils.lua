-- This is a set of common functions that are non-trivial in Lua
require 'lfs'

--------------------------------------------------------------------------------
-- SORT OF DEFAULTDICT
--------------------------------------------------------------------------------
-- Note that each element will get the same default value.
-- So if you make the default value a table,
-- each "empty" element in the returned table will effectively reference the same table.
-- If that's not what you want, you'll have to modify the function.
function defaultdict(default)
  local tbl = {}
  local mtbl = {}
  mtbl.__index = function(tbl, key)
    local val = rawget(tbl, key)
    return val or default
  end
  setmetatable(tbl, mtbl)
  return tbl
end


--------------------------------------------------------------------------------
-- CHECK EXISTENCE OF FILES / DIRECTORIES
--------------------------------------------------------------------------------
-- Taken from: http://stackoverflow.com/questions/4990990/lua-check-if-a-file-exists
-- no function checks for errors.
-- you should check for them
function is_file(name)
    if type(name)~="string" then return false end
    if not isDir(name) then
        return os.rename(name,name) and true or false
        -- note that the short evaluation is to
        -- return false instead of a possible nil
    end
    return false
end

function is_file_or_dir(name)
    if type(name)~="string" then return false end
    return os.rename(name, name) and true or false
end

function is_dir(name)
    if type(name)~="string" then return false end
    local cd = lfs.currentdir()
    local is = lfs.chdir(name) and true or false
    lfs.chdir(cd)
    return is
end


--------------------------------------------------------------------------------
-- MAKE DIRECTORY IF IT DOESN'T EXIST
--------------------------------------------------------------------------------
function make_dir_if_not_exists(name)
  if not is_dir(name) then
    lfs.mkdir(name)
  end
end

--------------------------------------------------------------------------------
-- LIST FILES/SUBDIRECTORIES IN A DIRECTORY
--------------------------------------------------------------------------------
-- Taken from: http://stackoverflow.com/questions/5303174/how-to-get-list-of-directories-in-lua
-- Lua implementation of PHP scandir function
-- When dir_only is true, only returns directories. 
-- File gives relative file path
-- function scandir(directory, dir_only)
--     local i, t, popen = 0, {}, io.popen
--     local pfile = nil
--     if dir_only then 
--     	pfile = popen('ls -d ' .. directory .. '/*/')
--     else
--     	pfile = popen('ls -a "'.. directory ..'"')
--     end
--     for filename in pfile:lines() do
--         i = i + 1
--         t[i] = filename
--     end
--     pfile:close()
--     return t
-- end
function scandir(directory, dir_only)
  local iter, dir_obj = lfs.dir(directory)
  local names = {} -- file or directory
  while true do
    local obj = iter(dir_obj)
    if (obj ~= '.') and (obj ~= '..') then  -- Skip these
      if obj == nil then break end
      if dir_only then
        if is_dir(directory .. '/' .. obj) then
          table.insert(names, obj)
        end
      else  -- Add it no matter what
        table.insert(names, obj)
      end
    end
  end
  return names
end


--------------------------------------------------------------------------------
-- SPLIT STRING
--------------------------------------------------------------------------------
-- Taken from here: http://stackoverflow.com/questions/1426954/split-string-in-lua
-- I should probably use regex though
function split(inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            t[i] = str
            i = i + 1
    end
    return t
end

--------------------------------------------------------------------------------
-- READ FILE
--------------------------------------------------------------------------------
-- http://lua-users.org/wiki/FileInputOutput
-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

--------------------------------------------------------------------------------
-- SLICE TABLE
--------------------------------------------------------------------------------
function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

--------------------------------------------------------------------------------
-- SIZE OF TABLE
-- If table isn't indexed by number, can't do #t
--------------------------------------------------------------------------------
function size_of_table(t)
  local count = 0
  for k,v in pairs(t) do
    count = count + 1
  end
  return count
end

--------------------------------------------------------------------------------
-- CONVERT NON-INTEGER INDEXED TABLE TO INTEGER IN ORDER TO SAVE TO CSV
-- If table isn't indexed by number, csvigo can't do ipairs
--------------------------------------------------------------------------------
function convert_table_for_csvigo(t)
  local new_table = {}
  for k,v in pairs(t) do
    if type(v) == 'boolean' then  -- Convert to 1 or 0
      if v then 
        v = 1
      else
        v = 0
      end
    end
    table.insert(new_table, {k, v})
  end
  return new_table
end


--------------------------------------------------------------------------------
-- INVERT TABLE: keys become values, values become keys
--------------------------------------------------------------------------------
function invert_table(t)
  local new_table = {}
  for k,v in pairs(t) do
    new_table[v] = k
  end
  return new_table
end


--------------------------------------------------------------------------------
-- TERNARY
-- Example usage: local epochs = ternary_op(opt.use_google_model, 2, 3)
--------------------------------------------------------------------------------
function ternary_op(condition, true_val, false_val)
  if condition then
    return true_val
  else
    return false_val
  end
end
