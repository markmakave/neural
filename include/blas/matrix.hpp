#pragma once

#include "blas/allocator.hpp"
#include "blas/vector.hpp"

namespace blas {

// For math
template <typename T, typename _alloc>
struct vector;

template <typename T, typename _alloc = allocator<T>>
struct matrix
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef value_type*         iterator;
    typedef const value_type*   const_iterator;
    typedef unsigned            size_type;      

protected:

    pointer _data;
    size_type _height, _width;

public:

// Constructors ///////////////////////////////////////////////////////////

    matrix()
    :   _data(nullptr),
        _height(0),
        _width(0)
    {}

    matrix(size_type height, size_type width)
    :   _height(height),
        _width(width)
    {
        _data = _allocate(size());
    }

    matrix(size_type width, size_type height, const_reference fillament) 
    :   _height(height),
        _width(width)
    {
        _data = _allocate(size());
        fill(fillament);
    }

    matrix(const matrix<T>& m)
    :   _height(m._height),
        _width(m._width)
    {
        _data = _allocate(size());

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] = m._data[i];
    }

    matrix(matrix&& m)
    :   _data(m._data),
        _height(m._height),
        _width(m._width)
    {
        m._data = nullptr;
        m._height = 0;
        m._width = 0;
    }
    
    template <typename _T>
    matrix(const matrix<_T>& m, std::function<T(const _T&)> f = [](const _T& x) { return static_cast<T>(x); })
    :   _height(m.height()),
        _width(m.width())
    {
        _data = _allocate(size());

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] = m.data()[i];
    }
    
// Destructor /////////////////////////////////////////////////////////////

    ~matrix()
    {
        _deallocate(_data);
    }

// Assignment operators ///////////////////////////////////////////////////

    matrix& 
    operator = (const matrix& m)
    {
        if (&m != this)
        {
            resize(m._width, m._height);

            #pragma omp parallel for
            for (size_type i = 0; i < size(); ++i)
                _data[i] = m._data[i];
        }
        return *this;
    }

    matrix& 
    operator = (matrix&& m)
    {
        if (&m != this)
        {
            _deallocate(_data);
            _width = m._width;
            _height = m._height;
            _data = m._data;
            m._data = nullptr;
        }
        return *this;
    }

// Resizing method ////////////////////////////////////////////////////////

    void
    resize(size_type width, size_type height)
    {
        if (_width != width or _height != height)
        {
            _deallocate(_data);
            _width = width;
            _height = height;
            _data = _allocate(size());
        }
    }

    void
    fill(const_reference fillament)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] = fillament;
    }

// Row getters ////////////////////////////////////////////////////////////

    pointer
    row(size_type index) 
    {
        return _data + index * _width;
    }

    const_pointer
    row(size_type index) const
    {
        return _data + index * _width;
    }

    pointer 
    operator [] (size_type index)
    {
        return row(index);
    }

    const_pointer 
    operator [] (size_type index) const 
    {
        return row(index);
    }

// Attributes getters /////////////////////////////////////////////////////

    size_type 
    size() const
    {
        return _width * _height;
    }

    size_type 
    width() const
    {
        return _width;
    }

    size_type 
    height() const
    {
        return _height;
    }

    pointer
    data() const
    {
        return _data;
    }

// Iterator methods ///////////////////////////////////////////////////////

    iterator
    begin()
    {
        return _data;
    }

    iterator
    end()
    {
        return _data + size();
    }

    const_iterator
    begin() const
    {
        return _data;
    }

    const_iterator
    end() const
    {
        return _data + size();
    }

// Binary operators ///////////////////////////////////////////////////////

    matrix
    operator + (const matrix& m) const
    {
        matrix result(*this);
        result += m;
        return result;
    }

    matrix
    operator - (const matrix& m) const
    {
        matrix result(*this);
        result -= m;
        return result;
    }

    matrix
    operator * (const_reference scalar) const
    {
        matrix result(*this);
        result *= scalar;
        return result;
    }

    matrix
    operator / (const_reference scalar) const
    {
        matrix result(*this);
        result /= scalar;
        return result;
    }

// Binary mutating operators //////////////////////////////////////////////

    matrix&
    operator += (const matrix& m)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] += m._data[i];
        return *this;
    }

    matrix&
    operator -= (const matrix& m)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] -= m._data[i];
        return *this;
    }

    matrix&
    operator *= (const_reference scalar)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] *= scalar;
        return *this;
    }

    matrix&
    operator /= (const_reference scalar)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] /= scalar;
        return *this;
    }

// Math ///////////////////////////////////////////////////////////////////

    vector<T, _alloc>
    operator * (const vector<T, _alloc>& v) const
    {
        assert(_width == v.size());

        vector<T, _alloc> result(_height);

        #pragma omp parallel for
        for (size_type i = 0; i < _height; ++i)
        {
            auto this_row = row(i);

            value_type sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (size_type j = 0; j < _width; ++j)
                sum += this_row[j] * v[j];
                
            result[i] = sum;
        }

        return result;
    }

    matrix
    operator * (const matrix& m) const
    {
        assert(_width == m._height);

        matrix result(_height, m._width);

        #pragma omp parallel for
        for (size_type i = 0; i < _height; ++i)
        {
            auto result_row = result[i];
            auto this_row = row(i);

            #pragma omp parallel for
            for (size_type j = 0; j < m._width; ++j)
            {
                value_type sum = 0;

                #pragma omp parallel for reduction(+:sum)
                for (size_type k = 0; k < _width; ++k)
                    sum += this_row[k] * m[k][j];

                result_row[j] = sum;
            }
        }

        return result;
    }

    matrix
    transpose()
    {
        matrix result(_width, _height);

        #pragma omp parallel for
        for (size_type y = 0; y < _height; ++y)
        {
            auto this_row = row(y);
            
            #pragma omp parallel for
            for (size_type x = 0; x < _width; ++x)
                result[x][y] = this_row[x];
        }

        return result;
    }

// Comparson //////////////////////////////////////////////////////////////

    bool
    operator == (const matrix& m) const
    {
        if (_width != m._width or _height != m._height) return false;

        for (size_type i = 0; i < size(); ++i)
            if (_data[i] != m._data[i]) return false;

        return true;
    }

    bool
    operator != (const matrix& m) const
    {
        return !((*this) == m);
    }

private:

    static
    pointer
    _allocate(size_type size)
    {
        return _alloc::allocate(size);
    }

    static
    void
    _deallocate(pointer ptr)
    {
        _alloc::deallocate(ptr);
    }

};

} // namespace lm
