#pragma once

#include <iostream>

#include "blas/allocator.hpp"
#include "blas/matrix.hpp"

namespace blas {

template <typename T, typename _alloc>
struct matrix;

template <typename T, typename _alloc = allocator<T>>
struct vector
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

    class reverse_iterator
    {
        iterator _it;

    public:

        reverse_iterator(iterator it)
        :   _it(it)
        {}

        reverse_iterator&
        operator++()
        {
            --_it;
            return *this;
        }

        reverse_iterator
        operator++(int)
        {
            reverse_iterator tmp(*this);
            operator++();
            return tmp;
        }

        reverse_iterator&
        operator--()
        {
            ++_it;
            return *this;
        }

        reverse_iterator
        operator--(int)
        {
            reverse_iterator tmp(*this);
            operator--();
            return tmp;
        }

        reference
        operator*()
        {
            return *_it;
        }

        pointer
        operator->()
        {
            return _it;
        }

        bool
        operator==(const reverse_iterator& rhs)
        {
            return _it == rhs._it;
        }

        bool
        operator!=(const reverse_iterator& rhs)
        {
            return _it != rhs._it;
        }
    };

    class const_reverse_iterator
    {
        const_iterator _it;

    public:

        const_reverse_iterator(const_iterator it)
        :   _it(it)
        {}

        const_reverse_iterator&
        operator++()
        {
            --_it;
            return *this;
        }

        const_reverse_iterator
        operator++(int)
        {
            const_reverse_iterator tmp(*this);
            operator++();
            return tmp;
        }

        const_reverse_iterator&
        operator--()
        {
            ++_it;
            return *this;
        }

        const_reverse_iterator
        operator--(int)
        {
            const_reverse_iterator tmp(*this);
            operator--();
            return tmp;
        }

        const_reference
        operator*()
        {
            return *_it;
        }

        const_pointer
        operator->()
        {
            return _it;
        }

        bool
        operator==(const const_reverse_iterator& rhs)
        {
            return _it == rhs._it;
        }

        bool
        operator!=(const const_reverse_iterator& rhs)
        {
            return _it != rhs._it;
        }
    };

protected:

    pointer _data;
    size_type _size, _capacity;

public:

// Constructors ///////////////////////////////////////////////////////////

    vector() 
    :   _data(nullptr),
        _size(0),
        _capacity(0)
    {}

    vector(size_type size) 
    :   _size(size),
        _capacity(size)
    {
        _data = _allocate(_capacity);
    }

    vector(const vector& m) 
    :   _size(m._size),
        _capacity(m._size)
    {
        _data = _allocate(_capacity);

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] = m._data[i];
    }

    vector(vector&& m) 
    :   _data(m._data),
        _size(m._size),
        _capacity(m._capacity)
    {
        m._data = nullptr;
    }
    
    template <typename _T>
    vector(const vector<_T>& m, std::function<T(const _T&)> f = [](const _T& v) { return static_cast<T>(v); })
    :   _size(m.size()),
        _capacity(m.size())
    {
        _data = _allocate(_capacity);

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] = f(m[i]);
    }
    
// Destructor /////////////////////////////////////////////////////////////

    ~vector()
    {
        _deallocate(_data);
    }

// Assignment operators ///////////////////////////////////////////////////

    vector& 
    operator = (const vector& m)
    {
        if (&m != this)
        {
            resize(m._size);

            #pragma omp parallel for
            for (size_type i = 0; i < size(); ++i)
                _data[i] = m._data[i];
        }
        return *this;
    }

    vector& 
    operator = (vector&& m)
    {
        if (&m != this)
        {
            _deallocate(_data);
            _size = m._size;
            _capacity = m._capacity;
            _data = m._data;
            m._data = nullptr;
        }
        return *this;
    }

// Resizing method ////////////////////////////////////////////////////////

    void
    resize(size_type size)
    {
        if (size <= _capacity)
        {
            _size = size;
        } else {
            _deallocate(_data);
            _size = size;
            _capacity = size;
            _data = _allocate(_capacity);
        }
    }

    void
    reserve(size_type size)
    {
        if (size > _capacity)
        {
            _deallocate(_data);
            _capacity = size;
            _data = _allocate(_capacity);
        }
    }

// Element etters ////////////////////////////////////////////////////////

    reference 
    operator [] (size_type index)
    {
        return _data[index];
    }

    const_reference 
    operator [] (size_type index) const 
    {
        return _data[index];
    }

    reference
    at(size_type index)
    {
        if (index >= _size)
            throw std::out_of_range("vector::at");
        return _data[index];
    }

    const_reference
    at(size_type index) const
    {
        if (index >= _size)
            throw std::out_of_range("vector::at");
        return _data[index];
    }

    reference
    front()
    {
        return _data[0];
    }

    const_reference
    front() const
    {
        return _data[0];
    }

    reference
    back()
    {
        return _data[_size - 1];
    }

    const_reference
    back() const
    {
        return _data[_size - 1];
    }

// Attributes getters /////////////////////////////////////////////////////

    size_type 
    size() const
    {
        return _size;
    }

    size_type 
    capacity() const
    {
        return _capacity;
    }

    pointer
    data() const
    {
        return _data;
    }

// Comparson //////////////////////////////////////////////////////////////

    void
    push(const_reference value)
    {
        if (_size == _capacity)
        {
            if (_capacity == 0)
                _capacity = 1;
            else
                _capacity *= 2;
                
            pointer _temp = _allocate(_capacity);
            
            #pragma omp parallel for
            for (size_type i = 0; i < _size; ++i)
                _temp[i] = _data[i];

            _deallocate(_data);
            _data = _temp;
        }

        _data[_size++] = value;
    }

    void
    push(T&& value)
    {
        if (_size == _capacity)
        {
            if (_capacity == 0) _capacity = 1;

            _capacity *= 2;
            pointer _temp = _allocate(_capacity);
            
            #pragma omp parallel for
            for (size_type i = 0; i < _size; ++i)
                _temp[i] = _data[i];

            _deallocate(_data);
            _data = _temp;
        }

        _data[_size++] = value;
    }

    value_type
    pop()
    {
        return _data[_size-- - 1];
    }

    void
    erase()
    {
        _deallocate(_data);
        _size = 0;
        _capacity = 0;
        _data = nullptr;
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

    const_iterator
    cbegin() const
    {
        return _data;
    }

    const_iterator
    cend() const
    {
        return _data + size();
    }

    reverse_iterator
    rbegin()
    {
        return reverse_iterator(end() - 1);
    }

    reverse_iterator
    rend()
    {
        return reverse_iterator(begin() - 1);
    }

    const_reverse_iterator
    rbegin() const
    {
        return const_reverse_iterator(end() - 1);
    }

    const_reverse_iterator
    rend() const
    {
        return const_reverse_iterator(begin() - 1);
    }

    const_reverse_iterator
    crbegin() const
    {
        return const_reverse_iterator(end() - 1);
    }

    const_reverse_iterator
    crend() const
    {
        return const_reverse_iterator(begin() - 1);
    }

// Comparson //////////////////////////////////////////////////////////////

    bool
    operator == (const vector& m) const
    {
        if (_size != m._size) return false;

        for (size_type i = 0; i < size(); ++i)
            if (_data[i] != m._data[i]) return false;

        return true;
    }

    bool
    operator != (const vector& m) const
    {
        return !((*this) == m);
    }

// Binary operators ///////////////////////////////////////////////////////

    vector
    operator + (const vector& m) const
    {
        vector result(*this);
        result += m;
        return result;
    }

    vector
    operator - (const vector& m) const
    {
        vector result(*this);
        result -= m;
        return result;
    }

    vector
    operator * (const vector& m) const
    {
        vector result(*this);
        result *= m;
        return result;
    }

    vector
    operator / (const vector& m) const
    {
        vector result(*this);
        result /= m;
        return result;
    }

    vector
    operator + (const_reference value) const
    {
        vector result(*this);
        result += value;
        return result;
    }

    vector
    operator - (const_reference value) const
    {
        vector result(*this);
        result -= value;
        return result;
    }

    vector
    operator * (const_reference value) const
    {
        vector result(*this);
        result *= value;
        return result;
    }

    vector
    operator / (const_reference value) const
    {
        vector result(*this);
        result /= value;
        return result;
    }

// Binary mutating operators //////////////////////////////////////////////

    vector&
    operator += (const vector& m)
    {
        if (_size != m._size)
            throw std::invalid_argument("vector::operator += : size mismatch");

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] += m._data[i];

        return *this;
    }

    vector&
    operator -= (const vector& m)
    {
        if (_size != m._size)
            throw std::invalid_argument("vector::operator -= : size mismatch");

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] -= m._data[i];

        return *this;
    }

    vector&
    operator *= (const vector& m)
    {
        if (_size != m._size)
            throw std::invalid_argument("vector::operator *= : size mismatch");

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] *= m._data[i];

        return *this;
    }

    vector&
    operator /= (const vector& m)
    {
        if (_size != m._size)
            throw std::invalid_argument("vector::operator /= : size mismatch");

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] /= m._data[i];

        return *this;
    }

    vector&
    operator += (const_reference value)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] += value;

        return *this;
    }

    vector&
    operator -= (const_reference value)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] -= value;
            
        return *this;
    }

    vector&
    operator *= (const_reference value)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] *= value;

        return *this;
    }

    vector&
    operator /= (const_reference value)
    {
        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            _data[i] /= value;

        return *this;
    }

// Unary operators /////////////////////////////////////////////////////////

    vector
    operator - () const
    {
        vector result(*this);

        #pragma omp parallel for
        for (size_type i = 0; i < size(); ++i)
            result._data[i] = -result._data[i];

        return result;
    }

    vector
    operator + () const
    {
        return *this;
    }

// Frien operators /////////////////////////////////////////////////////////

    friend
    vector
    operator + (const_reference value, const vector& v)
    {
        return v + value;
    }

    friend
    vector
    operator - (const_reference value, const vector& v)
    {
        return v - value;
    }

    friend
    vector
    operator * (const_reference value, const vector& v)
    {
        return v * value;
    }

// Math ///////////////////////////////////////////////////////////////////

    static 
    T
    inner_product(const vector& a, const vector& b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("dot: vectors must have the same size");

        T result = 0;

        #pragma omp parallel for reduction(+:result)
        for (size_t i = 0; i < a.size(); ++i)
            result += a[i] * b[i];

        return result;
    }

    static
    matrix<T>
    outer_product (const vector& a, const vector& b)
    {
        matrix<T> result(a.size(), b.size());

        #pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i)
        {
            #pragma omp parallel for
            for (size_t j = 0; j < b.size(); ++j)
            {
                result[i][j] = a[i] * b[j];
            }
        }
        
        return result;
    }

// Print //////////////////////////////////////////////////////////////////

    friend
    std::ostream&
    operator << (std::ostream& os, const vector& m)
    {
        os << "[";
        for (size_type i = 0; i < m.size(); ++i)
        {
            os << m[i];
            if (i != m.size() - 1) os << ", ";
        }
        os << "]";
        return os;
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
